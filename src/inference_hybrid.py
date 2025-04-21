import os
import sqlite3
import logging
import json
import torch
import time
import argparse
import concurrent.futures
from typing import List, Dict, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
from transformers import AutoModelForCausalLM, AutoTokenizer

os.makedirs("results", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results/validation_log.log'),
        logging.StreamHandler()
    ]
)

class QwenValidator:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            pad_token='<|endoftext|>',
            eos_token='<|im_end|>',
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        ).to(self.device)

        self.tokenizer.add_special_tokens({
            "additional_special_tokens": ["<|im_start|>", "<|im_end|>"]
        })
        self.model.resize_token_embeddings(len(self.tokenizer))

    def validate_sql(self, question: str, sql_query: str) -> dict:
        messages = [
            {
                "role": "system",
                "content": "You are a world-class SQL expert. Please evaluate whether the following SQL query semantically answers the given natural language question. You may include reasoning or analysis, but conclude your answer with either the word 'true' or 'false'."
            },
            {
                "role": "user",
                "content": f"Question: {question}\nSQL: {sql_query}"
            }
        ]
        formatted_input = self.format_chatml(messages)
        inputs = self.tokenizer(formatted_input, return_tensors="pt", truncation=True, max_length=8192).to(self.device)

        start_time = time.time()
        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=200,
            temperature=0.01,
            top_p=0.95,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        infer_time = time.time() - start_time

        response = self.tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip().lower()

        return {
            'prediction': self._parse_response(response),
            'response': response,
            'infer_time': infer_time
        }

    def format_chatml(self, messages: List[Dict]) -> str:
        return "".join([f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]).strip()

    def _parse_response(self, response: str) -> bool:
        response = response.lower().strip()
        if not response:
            return False
        return any(keyword in response.split()[-3:] for keyword in ['true', 'correct'])

class SQLEvaluator:
    def __init__(self, qwen_model_path: str):
        self.qwen = QwenValidator(qwen_model_path)
        self.database_folders = ['test_database', 'database']
        self.metrics = {
            'ea': {'true_labels': [], 'predictions': [], 'timings': []},
            'ensemble': {'true_labels': [], 'predictions': [], 'timings': []},
            'qwen': {'true_labels': [], 'predictions': [], 'timings': []},
            'category': {},
            'ea_category': {},
            'qwen_category': {},
            'ensemble_category': {},
            'ensemble_logic': {'agree': 0, 'ea_override': 0, 'qwen_override': 0}
        }
        self.timing = {'ea': 0.0, 'qwen': 0.0, 'ensemble': 0.0}
        self.error_records = []
        self.misclassified = {k: [] for k in ['qwen', 'ea', 'ensemble']}

    def _execute_query(self, query: str, db_id: str) -> Tuple[frozenset, str]:
        for folder in self.database_folders:
            db_path = os.path.join(folder, db_id, f'{db_id}.sqlite')
            if os.path.exists(db_path):
                try:
                    with sqlite3.connect(db_path) as conn:
                        conn.row_factory = sqlite3.Row
                        cursor = conn.cursor()
                        cursor.execute(query)
                        results = cursor.fetchall()
                        return self._normalize_results(results), None
                except Exception as e:
                    return None, str(e)
        return None, "Database not found"

    def _normalize_results(self, results: list) -> frozenset:
        if not results:
            return frozenset()
        return frozenset(
            frozenset(sorted(row.items())) if isinstance(row, dict) else frozenset(sorted(row))
            for row in results
        )

    def _process_item(self, item: Dict) -> Tuple[int, int]:
        category = item.get('category', 'Unknown')

        with concurrent.futures.ThreadPoolExecutor() as executor:
            ea_future = executor.submit(self._run_ea, item)
            qwen_future = executor.submit(self.qwen.validate_sql, item['question'], item['query'])

            ea_pred, ea_time = ea_future.result()
            qwen_result = qwen_future.result()

        if ea_pred is None:
            return None, None

        qwen_pred = qwen_result['prediction']
        ensemble_pred = self._get_ensemble_prediction(ea_pred, qwen_pred)

        ensemble_time = max(ea_time, qwen_result['infer_time'])

        self._update_metrics(item, category, ea_pred, qwen_pred, ensemble_pred, ea_time, qwen_result['infer_time'], ensemble_time)

        return ea_pred, ensemble_pred

    def _get_ensemble_prediction(self, ea_pred, qwen_pred):
        if ea_pred == qwen_pred:
            self.metrics['ensemble_logic']['agree'] += 1
            return ea_pred
        elif ea_pred == 1 and qwen_pred == 0:
            self.metrics['ensemble_logic']['qwen_override'] += 1
            return qwen_pred
        elif ea_pred == 0 and qwen_pred == 1:
            self.metrics['ensemble_logic']['ea_override'] += 1
            return ea_pred

    def _update_metrics(self, item, category, ea_pred, qwen_pred, ensemble_pred, ea_time, qwen_time, ensemble_time):
        self.timing['ea'] += ea_time
        self.timing['qwen'] += qwen_time
        self.timing['ensemble'] += ensemble_time

        for system, pred, t in [('ea', ea_pred, ea_time), ('qwen', qwen_pred, qwen_time), ('ensemble', ensemble_pred, ensemble_time)]:
            if pred is not None:
                self.metrics[system]['true_labels'].append(item['correctness'])
                self.metrics[system]['predictions'].append(pred)
                self.metrics[system]['timings'].append(t)

                if category not in self.metrics[f'{system}_category']:
                    self.metrics[f'{system}_category'][category] = {'true_labels': [], 'predictions': [], 'timings': []}
                self.metrics[f'{system}_category'][category]['true_labels'].append(item['correctness'])
                self.metrics[f'{system}_category'][category]['predictions'].append(pred)
                self.metrics[f'{system}_category'][category]['timings'].append(t)

                if pred != item['correctness']:
                    self.misclassified[system].append({**item, 'predicted': pred, 'system': system})

    def _run_ea(self, item: Dict) -> Tuple[int, float]:
        start = time.time()
        pred_result, pred_error = self._execute_query(item['query'], item['db_id'])
        gold_result, gold_error = self._execute_query(item['gold_parse'], item['db_id'])
        ea_time = time.time() - start

        if pred_error or gold_error:
            self.error_records.append({**item, 'predicted_error': pred_error, 'gold_error': gold_error})
            return None, ea_time

        return int(pred_result == gold_result), ea_time

    def evaluate(self, data: List[Dict]):
        for item in data:
            self._process_item(item)
        self._report_metrics()

    def _report_metrics(self):
        for name in ['ea', 'qwen', 'ensemble']:
            if self.metrics[name]['true_labels']:
                logging.info(f"\n🔹 {name.upper()} Overall Metrics:")
                metrics = self._calculate_metrics(self.metrics[name])
                self._log_metrics(metrics)
                avg_time = sum(self.metrics[name]['timings']) / len(self.metrics[name]['timings']) * 1000
                logging.info(f"Average Inference Time per Record: {avg_time:.2f} ms")

        for name in ['qwen', 'ea', 'ensemble']:
            logging.info(f"\n🔹 Category-wise {name.upper()} Metrics:")
            for cat, data in self.metrics[f'{name}_category'].items():
                logging.info(f"\nCategory: {cat}")
                self._log_metrics(self._calculate_metrics(data))
                avg_time = sum(data['timings']) / len(data['timings']) * 1000
                logging.info(f"Average Inference Time per Record: {avg_time:.2f} ms")

        for k, v in self.timing.items():
            logging.info(f"Total {k.upper()} Inference Time: {v:.2f} seconds")

        for k, v in self.metrics['ensemble_logic'].items():
            total = sum(self.metrics['ensemble_logic'].values())
            logging.info(f"{k.replace('_', ' ').title()}: {v} ({(v/total*100):.2f}%)")

        for system in ['qwen', 'ea', 'ensemble']:
            with open(f'results/misclassified_{system}.json', 'w') as f:
                json.dump(self.misclassified[system], f, indent=2)

    def _calculate_metrics(self, data: Dict) -> Dict:
        return {
            'accuracy': accuracy_score(data['true_labels'], data['predictions']),
            'precision': precision_score(data['true_labels'], data['predictions'], zero_division=1),
            'recall': recall_score(data['true_labels'], data['predictions'], zero_division=1),
            'f1': f1_score(data['true_labels'], data['predictions'], zero_division=1),
            'classification_report': classification_report(data['true_labels'], data['predictions'])
        }

    def _log_metrics(self, metrics: Dict):
        logging.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logging.info(f"Precision: {metrics['precision']:.4f}")
        logging.info(f"Recall: {metrics['recall']:.4f}")
        logging.info(f"F1 Score: {metrics['f1']:.4f}")
        logging.info("\nClassification Report:\n" + metrics['classification_report'])

def load_data(file_path: str) -> List[Dict]:
    with open(file_path) as f:
        return json.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SQL correctness using Qwen2.5 and execution accuracy.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to Qwen2.5 model directory")
    parser.add_argument("--data_path", type=str, required=True, help="Path to JSON dataset")
    args = parser.parse_args()

    start_time = time.time()
    evaluator = SQLEvaluator(qwen_model_path=args.model_path)
    data = load_data(args.data_path)
    evaluator.evaluate(data)
    logging.info(f"\nTotal Inference Time: {time.time() - start_time:.2f} seconds")
