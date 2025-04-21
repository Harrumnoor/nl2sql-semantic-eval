import os
import json
import time
import torch
import argparse
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

os.makedirs("results", exist_ok=True)

class SQLCorrectnessDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        combined_input = f"{item['question']} [SEP] {item['query']}"
        inputs = self.tokenizer.encode_plus(
            combined_input,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(item['correctness'], dtype=torch.long),
            'query_data': item
        }

def custom_collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.tensor([item['labels'] for item in batch])
    query_data = [item['query_data'] for item in batch]
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels, 'query_data': query_data}

def load_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def run_inference(test_data_path, model_path, batch_size=8):
    test_data = load_data(test_data_path)

    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    dataset = SQLCorrectnessDataset(test_data, tokenizer, max_len=512)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate_fn)

    all_preds, all_labels = [], []
    categories, subcategories = [], []
    inference_times = []
    misclassified = []

    cat_times = defaultdict(list)

    start_time = time.time()
    with torch.no_grad():
        for batch in loader:
            inputs = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            query_batch = batch['query_data']

            batch_start = time.time()
            outputs = model(inputs, attention_mask=masks)
            batch_time = time.time() - batch_start
            per_record_time = batch_time / len(query_batch)
            inference_times.extend([per_record_time] * len(query_batch))

            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            for i, item in enumerate(query_batch):
                cat = item['category']
                subcat = item['sub-category']
                pred = preds[i].item()
                true = labels[i].item()
                categories.append(cat)
                subcategories.append(subcat)
                cat_times[cat].append(per_record_time)

                if pred != true:
                    item['predicted'] = pred
                    misclassified.append(item)

    total_time = time.time() - start_time
    avg_time = np.mean(inference_times)

    with open("results/misclassified_codebert.json", "w") as f:
        json.dump(misclassified, f, indent=2)

    print(f"\n Total Inference Time: {total_time:.2f} seconds")
    print(f"Average Inference Time per Record: {avg_time:.4f} seconds")

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"\n Overall Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Incorrect', 'Correct']))

    print("\n Per-category Metrics:")
    for category in sorted(set(categories)):
        cat_labels = [l for l, c in zip(all_labels, categories) if c == category]
        cat_preds = [p for p, c in zip(all_preds, categories) if c == category]
        if not cat_labels:
            continue
        print(f"\nCategory: {category}")
        print(f"Samples: {len(cat_labels)}")
        print(f"Accuracy: {accuracy_score(cat_labels, cat_preds):.4f}")
        print(f"Precision: {precision_score(cat_labels, cat_preds):.4f}")
        print(f"Recall: {recall_score(cat_labels, cat_preds):.4f}")
        print(f"F1 Score: {f1_score(cat_labels, cat_preds):.4f}")
        print(f"Avg Inference Time: {np.mean(cat_times[category]):.4f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for CodeBERT-based SQL Correctness Classifier")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned CodeBERT model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to evaluation dataset")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    args = parser.parse_args()

    run_inference(args.data_path, args.model_path, args.batch_size)
