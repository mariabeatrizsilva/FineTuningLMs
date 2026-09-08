# Fine-Tuning Language Models (FineTuningLMs)  (done as part of Eunsol Choi's NLP Course at NYU)

This repository contains code, notebooks, and scripts dedicated to fine-tuning large language models (LLMs) and smaller pre-trained language models for various Natural Language Processing (NLP) tasks.
---

## 📌 Features

- **Parameter-Efficient Fine-Tuning (PEFT):** Implementations using LoRA, QLoRA, and prefix tuning to train models efficiently with limited GPU memory.
- **Instruction Tuning & Supervised Fine-Tuning (SFT):** Scripts for fine-tuning open-source models (e.g., Llama, Mistral, Gemma, or BERT/RoBERTa variants) on custom dataset formats.
- **Data Preprocessing:** Utilities and scripts to clean, tokenize, and format raw datasets into prompt templates.
- **Evaluation & Inference:** Pipeline examples to run evaluation metrics and test model outputs post-training.
---
## 🛠️ Installation & Setup
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/mariabeatrizsilva/FineTuningLMs.git](https://github.com/mariabeatrizsilva/FineTuningLMs.git)
   cd FineTuningLMs
```

2. **Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```


3. **Install dependencies:**
```bash
pip install -r requirements.txt

```



*(Note: Ensure you have CUDA installed if training on an NVIDIA GPU).*

---

## 📂 Repository Structure

```text
FineTuningLMs/
├── data/               # Raw and preprocessed datasets (if applicable)
├── notebooks/          # Exploratory analysis and step-by-step Jupyter notebooks
├── scripts/            # Python scripts for training, fine-tuning, and evaluation
├── checkpoints/        # Saved adapter weights/model checkpoints (git-ignored)
├── requirements.txt    # Required Python packages
└── README.md           # Project documentation

```

---

## 🚀 Usage

### 1. Preprocessing Data

Run data preparation scripts to structure training sets into the expected prompt formats:

```bash
python scripts/preprocess.py --input_path data/raw.json --output_path data/processed.json

```

### 2. Fine-Tuning

Execute the fine-tuning training loop:

```bash
python scripts/train.py --model_name "meta-llama/Llama-2-7b-hf" --dataset_path data/processed.json

```

### 3. Inference

Run model inference using fine-tuned adapters/weights:

```bash
python scripts/inference.py --checkpoint_path checkpoints/best_model

```

---

## 📦 Requirements

Common dependencies used across this project:

* `torch`
* `transformers`
* `datasets`
* `peft`
* `bitsandbytes`
* `trl`
* `accelerate`

