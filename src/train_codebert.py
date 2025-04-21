import os
import json
import torch
import torch.nn as nn
import optuna
import argparse
import numpy as np
from tqdm import tqdm
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW, get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

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
            'labels': torch.tensor(item['correctness'], dtype=torch.long)
        }

def load_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def compute_metrics(preds, labels):
    preds = torch.argmax(preds, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=1),
        "recall": recall_score(labels, preds, zero_division=1),
        "f1": f1_score(labels, preds, zero_division=1)
    }

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_preds.append(outputs.logits)
            all_labels.append(labels)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    return compute_metrics(all_preds, all_labels)

def objective(trial, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    model = RobertaForSequenceClassification.from_pretrained(args.model_name, num_labels=2).to(device)

    dropout_prob = trial.suggest_float("dropout_prob", 0.1, 0.4, step=0.1)
    lr_classifier = trial.suggest_float("lr_classifier", 1e-4, 1e-3, log=True)
    lr_lower_layers = trial.suggest_float("lr_lower_layers", 1e-5, 1e-4, log=True)
    lr_higher_layers = trial.suggest_float("lr_higher_layers", 1e-5, 1e-4, log=True)

    model.classifier.dropout = nn.Dropout(dropout_prob)

    optimizer_grouped_parameters = [
        {"params": model.roberta.encoder.layer[:6].parameters(), "lr": lr_lower_layers},
        {"params": model.roberta.encoder.layer[6:].parameters(), "lr": lr_higher_layers},
        {"params": model.classifier.parameters(), "lr": lr_classifier},
    ]
    optimizer = AdamW(optimizer_grouped_parameters)

    train_data = load_data(args.train_path)
    train_data, val_data = train_test_split(train_data, test_size=0.2, stratify=[d["correctness"] for d in train_data])

    train_set = SQLCorrectnessDataset(train_data, tokenizer, args.max_len)
    val_set = SQLCorrectnessDataset(val_data, tokenizer, args.max_len)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    total_steps = len(train_loader) * args.epochs
    scheduler = get_scheduler("linear", optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    best_score = 0

    for epoch in range(args.epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        metrics = evaluate_model(model, val_loader, device)
        avg_score = np.mean(list(metrics.values()))
        best_score = max(best_score, avg_score)

        if avg_score < 0.5:
            break

    return best_score

def main(args):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, args), n_trials=args.trials)

    print("\n Best Trial Results:")
    print(json.dumps(study.best_params, indent=2))

    with open("results/best_hyperparams_codebert.json", "w") as f:
        json.dump(study.best_params, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune CodeBERT for SQL semantic correctness with Optuna.")
    parser.add_argument("--model_name", type=str, default="microsoft/codebert-base", help="Base model name")
    parser.add_argument("--train_path", type=str, required=True, help="Path to training dataset (JSON)")
    parser.add_argument("--max_len", type=int, default=512, help="Max token length")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Max number of epochs")
    parser.add_argument("--trials", type=int, default=10, help="Number of Optuna trials")
    args = parser.parse_args()
    main(args)
