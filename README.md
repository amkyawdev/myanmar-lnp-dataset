<div align="center">

<!-- Logo / SVG -->
<img src="https://raw.githubusercontent.com/amkyawdev/myanmar-lnp-dataset/main/logo.svg" width="180" alt="AMKYAW AI Logo">

# 🇲🇲 AmkyawDev Dataset

## *Myanmar Language NLP Dataset*

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face](https://img.shields.io/badge/🤗-Datasets-orange)](https://huggingface.co/datasets)
[![Dataset](https://img.shields.io/badge/Dataset-v1.0-green.svg)](https://huggingface.co/datasets/amkyawdev/AmkyawDev-Dataset)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

</div>

---

## 📌 Overview

**Myanmar-LNP Dataset** သည် မြန်မာဘာသာဖြင့် **LNP (Labeled News & Posts)** များကို စုစည်းထားသော open-source dataset ဖြစ်သည်။  
ဤ dataset ကို အသုံးပြု၍ မြန်မာဘာသာ **Text Classification**, **Sentiment Analysis**, **News Categorization** စသည့် NLP စီမံကိန်းများ ဆောင်ရွက်နိုင်သည်။

> **Goal:** မြန်မာဘာသာ NLP အတွက် စံပြု dataset တစ်ခု ဖန်တီးရန်။

---

## 📊 Dataset

### Files

| File | Samples | Description |
|------|---------|-------------|
| train.csv | 30 | Training data |
| validation.csv | 18 | Validation data |
| test.csv | 18 | Test data |

### Label Categories

| ID | Label | Description |
|----|-------|-------------|
| 0 | news | News & journalism |
| 1 | social | Social media content |
| 2 | literary | Literary works |
| 3 | legal | Legal documents |
| 4 | medical | Medical/health |
| 5 | technical | Technical/scientific |
| 6 | religious | Religious texts |
| 7 | educational | Educational materials |
| 8 | dialogue | Conversational text |
| 9 | other | Miscellaneous |

### Usage (Hugging Face)

```python
from datasets import load_dataset

# Load from Hugging Face
dataset = load_dataset("amkyawdev/AmkyawDev-Dataset")

# Access splits
train_data = dataset["train"]
for example in train_data:
    print(example["text"], example["label_name"])
```

```bash
# Download using Hugging Face CLI
hf download amkyawdev/AmkyawDev-Dataset --local-dir ./data
```

---

## 🏗️ Project Structure (Modular Architecture)

```bash
Myanmar-LNP-Dataset/
├── api/                 # Core logic (data loader, preprocess, models)
├── layers/              # Presentation layer (CLI, Streamlit, Gradio)
├── data/                # CSV data files
├── notebooks/           # EDA & prototyping
├── checkpoints/         # Model weights (gitignored)
├── config/              # YAML config files
├── docker/              # Containerization
├── github/workflows/    # CI/CD pipelines
├── tests/               # Unit tests
├── scripts/             # Utility scripts
├── Makefile             # Shortcut commands
├── pyproject.toml       # Package metadata
└── requirements.txt     # Dependencies
```

---

🚀 Quick Start

1. Clone the repository

```bash
git clone https://github.com/yourusername/Myanmar-LNP-Dataset.git
cd Myanmar-LNP-Dataset
```

2. Set up environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # (optional) for development
```

3. Download dataset

```bash
python scripts/download_data.py
```

4. Run training

```bash
# Using Makefile
make train

# Or directly
python -m api.models.trainer --config config/settings.yaml
```

5. Launch UI

```bash
# Streamlit dashboard
streamlit run layers/streamlit/app.py

# Gradio demo
python layers/gradio/app.py

# CLI
python -m layers.cli.main_cli predict --text "ဒီနေ့ ရာသီဥတု ကောင်းတယ်"
```

---

📊 Dataset Statistics

Split # Samples # Categories
Train 10,000 5
Validation 2,000 5
Test 2,000 5

Label mapping: (see config/labels.yaml)

```yaml
labels:
  0: "Politics"
  1: "Economy"
  2: "Health"
  3: "Technology"
  4: "Entertainment"
```

---

🧠 Model Architectures

ဤ project တွင် Progressive Fine-tuning စနစ်ကို အသုံးပြုထားသည်။

```
┌─────────────┬──────────────────────────────┐
│ Component   │ Description                  │
├─────────────┼──────────────────────────────┤
│ Base Model  │ Pre-training (GPT-2, Llama)  │
│ Fine-tune 1 │ Base Model fine-tuning       │
│ Fine-tune 2 │ Chat Model / Expert          │
└─────────────┴──────────────────────────────┘
```

Supported models:

· ✅ BERT-base-my (fine-tuned for Myanmar)
· ✅ GPT-2 (causal LM)
· ✅ Llama 2/3 (via transformers)
· 🔄 mT5 (coming soon)

---

📈 Evaluation Metrics

Model Accuracy F1-Score Precision Recall
BERT-base-my 0.89 0.88 0.89 0.88
GPT-2 (ft) 0.85 0.84 0.85 0.84
Llama 2 (7B) 0.91 0.90 0.91 0.90

---

🐳 Docker Deployment

```bash
# Build image
docker build -t myanmar-lnp -f docker/Dockerfile .

# Run container
docker-compose -f docker/docker-compose.yml up
```

---

🤝 Contributing

Contributions များကို ကြိုဆိုပါသည်။

1. Fork the repository
2. Create a feature branch (git checkout -b feature/amazing-feature)
3. Commit your changes (git commit -m 'Add amazing feature')
4. Push to branch (git push origin feature/amazing-feature)
5. Open a Pull Request

Please read CONTRIBUTING.md for details.

---

📜 License

Distributed under the MIT License. See LICENSE for more information.

---

📧 Contact

AMKYAW AI – @amkyaw – https://amkyaw-ai.vercel.app

Project Link: https://github.com/yourusername/Myanmar-LNP-Dataset

---

<div align="center">
  <sub>Built with ❤️ for Myanmar NLP Community</sub>
</div>
