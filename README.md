# nl2sql-semantic-eval
A unified repository for evaluating the **semantic correctness** of SQL queries generated from natural‑language questions.  
Combines:

-  **Execution Accuracy (EA)** — compares predicted vs. gold SQL results  
-  **CodeBERT** — fine‑tuned classifier for SQL correctness  
-  **Qwen2.5** — LLM‑based reasoning evaluation  
-  **Ensemble Mode** — merges EA & Qwen2.5 decisions
  
  **Installation** 
1. Clone the repo
     git clone https://github.com/Harrumnoor/nl2sql-semantic-eval
     cd nl2sql-semantic-eval
2. Create & activate a virtualenv
     python3 -m venv venv
     source venv/bin/activate     # macOS/Linux
     venv\Scripts\activate        # Windows
3. Install dependencies
     pip install -r requirements.txt
4. Configure paths
     export DATA_DIR=./data
     export TRAIN_PATH=$DATA_DIR/train.json
     export TEST_PATH=$DATA_DIR/test.json

   **Usage**
   **CodeBERT Training:**
   python src/train_codebert.py \
  --train_path $TRAIN_PATH \
  --model_name microsoft/codebert-base \
  --max_len 512 \
  --batch_size 8 \
  --epochs 10 \
  --trials 10
   
   **CodeBERT Evaluation:**
   python src/inference_codebert.py \
  --model_path $CODEBERT_MODEL \
  --data_path $TEST_PATH \
  --batch_size 8

   **Evaluating Hybrid(Qwen2.5 + EA)**
   python src/inference_qwen.py \
  --model_path $QWEN_MODEL \
  --data_path $TEST_PATH

   





