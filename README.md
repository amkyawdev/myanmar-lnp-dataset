# 🇲🇲 Myanmar LNP Dataset

<p align="center">
  <img src="Logo.svg" alt="Myanmar LNP Dataset Logo" width="400"/>
</p>

```
   ██████╗██████╗ ██████╗  ██████╗ ██╗  ██╗██╗███╗   ███╗███████╗    ██████╗  ██████╗  ██████╗ 
  ██╔════╝██╔══██╗██╔══██╗██╔═══██╗██║  ██║██║████╗ ████║██╔════╝  ██╔══██╗██╔═══██╗██╔══██╗
  ██║     ██████╔╝██████╔╝██║   ██║███████║██║██╔██╔██║█████╗    ██║  ██║██║   ██║█████╔╝ 
  ██║     ██╔══██╗██╔══██╗██║   ██║██╔══██║██║██║╚██╔╝██╔══╝    ██║  ██║██║   ██║██╔══██╗
  ╚██████╗██║  ██║██║  ██║╚██████╔╝██║  ██║██║██║ ╚═╝███████╗██████╔╝╚██████╔╝██║  ██║
   ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝╚═╝     ╚═════╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝
```

```
  _   _                  _____ _           _          
 | | | | __ _  ___  _ __|_   _| |__   ___ | | __ ___
 | |_| |/ _` |/ _ \| '_ \ | | | '_ \ / _ \| |/ // _ \
 |  _  | (_| | (_) | | | | | | | |_) |   <|  __/
 |_| |_|\__,_|\___/|_| |_|___||_.__/ \___/|_|\__\___|
                                                  
 _   _               _   _                       _    
| | | | _____  __ __| | |__   ___  _   _ _ __ __| | __
| |_| |/ _ \ \ / / _` | '_ \ / _ \| | | | '__/ _` |/ /
|  _  |  __/\ V / (_| | |_) |  __/| |_| | | | (_|   < 
|_| |_|\___| \_/ \__,_|_.__/ \___| \__,_|_|  \__,_|_|\_\
                                                  
██████╗  ██████╗  ██████╗  ██████╗ ████████╗███████╗███████╗██████╗ 
██╔══██╗██╔═══██╗██╔══██╗██╔══██╗╚══██╔══╝██╔════╝██╔════╝██╔══██╗
██║  ██║██║   ██║██████╔╝██║  ██║   ██║   █████╗  █████╗ ██████╔╝
██║  ██║██║   ██║██╔══██╗██║  ██║   ██║   ██╔══╝  ██╔══╝ ██╔══██╗
██████╔╝╚██████╔╝██║  ██║╚██████╔╝   ██║   ███████╗███████╗██║  ██║
╚═════╝  ╚═════╝ ╚═╝  ╚═╝ ╚═════╝    ╚═╝   ╚═════╝╚═════╝╚═╝  ╚═╝

██████╗  ██████╗  ██████╗  █ ██╗██╗███████╗██████╗ ███████╗██████╗ 
██╔══██╗██╔═══██╗██╔══██╗ ██║╚═╝██║██╔════╝██╔══██╗██╔════╝██╔══██╗
██████╔╝██║   ██║██████╔╝ ██║ ██║██║█████╗  ██████╔╝█████╗  ██████╔╝
██╔══██╗██║   ██║██╔══██╗ ██║ ██║██║██╔══╝  ██╔══██╗██╔══╝  ██╔══██╗
██║  ██║╚██████╔╝██║  ██║ ██████╔╝██║███████╗██████╔╝███████╗██║  ██║
╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═╝╚══════╝╚═════╝ ╚══════╝╚═╝  ╚═╝
```

**AmkyawDev** - A comprehensive toolkit for Myanmar (Burmese) language Natural Language Processing

## Features

- **Data Loading**: Support for JSONL, CSV, and Parquet formats
- **Preprocessing**: Unicode normalization, Zawgyi-to-Unicode conversion, tokenization
- **Augmentation**: Synonym replacement, random insertion/swap
- **Vectorization**: TF-IDF, Word2Vec, and BERT-based features
- **Classification**: PyTorch neural networks, sklearn classifiers
- **CLI**: Command-line tools for all operations
- **Web UI**: Streamlit and Gradio interfaces

## Installation

### From Source

```bash
# Clone repository
git clone https://github.com/<your-repo>/myanmar-lnp-dataset.git
cd myanmar-lnp-dataset

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### With Docker

```bash
docker build -t myanmar-lnp .
docker run -it myanmar-lnp
```

## Quick Start

### Using CLI

```bash
# Preprocess data
python -m layers.cli.main_cli preprocess input.jsonl output.jsonl

# Train model
python -m layers.cli.main_cli train train.jsonl test.jsonl

# Show stats
python -m layers.cli.main_cli stats data.jsonl
```

### Using Streamlit

```bash
streamlit run layers/streamlit/app.py
```

### Using Python

```python
from api.data_loader import load_data
from api.preprocess import MyanmarPreprocessor
from api.vectorizer import create_vectorizer

# Load data
df = load_data("data.jsonl")

# Preprocess
preprocessor = MyanmarPreprocessor()
df["text"] = preprocessor.transform(df["text"].tolist())

# Vectorize
vectorizer = create_vectorizer("tfidf")
X = vectorizer.fit_transform(df["text"].tolist())
```

## Project Structure

```
myanmar-lnp-dataset/
├── api/                    # Core logic
│   ├── data_loader.py      # Data loading
│   ├── preprocess.py    # Text preprocessing
│   ├── augment.py      # Data augmentation
│   ├── vectorizer.py   # Feature extraction
│   └── models/        # PyTorch models
├── layers/              # Presentation
│   ├── cli/          # Command-line
│   ├── streamlit/     # Streamlit UI
│   └── gradio/       # Gradio UI
├── config/            # Configuration
├── data/              # Data storage
├── tests/             # Tests
└── scripts/          # Utility scripts
```

## Configuration

Edit `config/settings.yaml` to customize:
- Data paths
- Model hyperparameters
- Preprocessing options

Edit `config/labels.yaml` to add custom label categories.

## Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v

# Format code
make format

# Lint
make lint
```

## License

MIT License - See LICENSE file.

## Contributing

See CONTRIBUTING.md for guidelines.