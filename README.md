# nl2sql-semantic-eval

A unified repository for evaluating the **semantic correctness** of SQL queries generated from natural‑language questions.  

Combines:
- **Execution Accuracy (EA)** — compares predicted vs. gold SQL results  
- **CodeBERT** — fine‑tuned classifier for SQL correctness  
- **Qwen2.5** — LLM‑based reasoning evaluation  
- **Ensemble Mode** — merges EA & Qwen2.5 decisions  

---

## ⚙ Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/Harrumnoor/nl2sql-semantic-eval
   cd nl2sql-semantic-eval
   ```

2. **Create & activate a virtualenv**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate      # macOS/Linux
   venv\Scripts\activate       # Windows
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure paths** (export as env vars or substitute directly)  
   ```bash
   export DATA_DIR=./data
   export TRAIN_PATH=$DATA_DIR/train.json
   export TEST_PATH=$DATA_DIR/test.json
   ```

---

##  Usage

### 1. Train CodeBERT (with Optuna)
```bash
python src/train_codebert.py \
  --train_path $TRAIN_PATH \
  --model_name microsoft/codebert-base \
  --max_len 512 \
  --batch_size 8 \
  --epochs 10 \
  --trials 10
```

### 2. Evaluate with CodeBERT
```bash
python src/inference_codebert.py \
  --model_path $CODEBERT_MODEL \
  --data_path $TEST_PATH \
  --batch_size 8
```

### 3. Hybrid Evaluation (Qwen2.5 + EA Ensemble)
```bash
python src/inference_qwen.py \
  --model_path $QWEN_MODEL \
  --data_path $TEST_PATH
```

---

##  Dataset Format

Each JSON record must include:
```json
{
  "question":   "Your natural‑language question",
  "query":      "Predicted SQL query",
  "gold_parse": "Gold SQL query",
  "correctness": 0 | 1,
  "db_id":      "database identifier",
  "category":   "Category code",
}
```

---

##  Repository Structure

```
nl2sql-semantic-eval/
├── src/
│   ├── train_codebert.py
│   ├── inference_codebert.py
│   └── inference_qwen.py
├── data/
│   ├── train.json
│   └── test.json
├── checkpoints/            # Qwen2.5 model
├── trained_model_best/     # CodeBERT model
├── results/                # logs & misclassified outputs
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE).


