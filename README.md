# BERT-based Design Mining for Software Architecture Issues

**"An Algorithmic Approach to Understanding the Impact of Design Work in Software Projects"** by Steven Morgan, Binghamton University, 2026.

## Overview

This project implements Research Question 1 from the dissertation:

> "Can design mining be performed on tickets collected from JIRA using transformers to outperform the state-of-the-art?"

The approach fine-tunes BERT (Bidirectional Encoder Representations from Transformers) for binary classification of software engineering tickets as either **'design'** or **'general'** (non-design) related.

## Key Features

- **Multi-Architecture Support**: Fine-tune BERT, DistilBERT, or RoBERTa for design issue identification
- **Transfer Learning Pipeline**: Three-stage pipeline from Stack Overflow to TAWOS dataset
- **TAWOS Database Integration**: Direct MySQL connection to TAWOS database for live data fetching
- **Comprehensive Preprocessing**: Software engineering-specific text cleaning with configurable stopwords
- **Baseline Comparisons**: Compare against Logistic Regression and SVM
- **Hyperparameter Tuning**: Grid search with detailed analysis and visualizations
- **Metrics**: Accuracy, Precision, Recall, F1, AUC with statistical significance testing

## Model Architecture (from Section 4.2.6)

```
Input Text → BERT Tokenizer → BERT Encoder (12 layers) → [CLS] Token
    ↓
Dropout Layer (0.1-0.3) → Dense Layer (768→2) → Softmax → P(Design)/P(Non-design)
```

Key parameters:
- **Max Length**: 512 tokens (captures JIRA title, description, comments)
- **Learning Rate**: 2e-5 (minimizes effect on pre-trained knowledge)
- **Batch Size**: 16
- **Epochs**: 3-5 for verification

## Installation

```bash
# Clone or download this repository
cd design-detection-and-defect-improvement

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### GPU Support (Recommended)

For faster training, install PyTorch with CUDA support:

```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Quick Start

### 1. Demo Mode (Sample Data)

Run a quick training demonstration with synthetic data:

```bash
python train_design_classifier.py --mode demo --epochs 2
```

This will:
- Generate 500 sample tickets
- Train the BERT model
- Compare with baseline models (Logistic Regression, SVM)
- Save the model to `./model_output`

### 2. Full Training with Real Data

#### Download Datasets

| Dataset | Description | Location |
|---------|-------------|----------|
| Stack Overflow | 230,002 labeled discussions | https://zenodo.org/records/4010209 |
| TAWOS | 458,232 JIRA issues | https://github.com/SOLAR-group/TAWOS |

```bash
# Create data directory
mkdir -p data/stackoverflow data/tawos

# Download Stack Overflow dataset
# Visit https://zenodo.org/records/4010209 and download

# Clone TAWOS dataset
git clone https://github.com/SOLAR-group/TAWOS.git data/tawos
```

#### Run Training

```bash
python train_design_classifier.py --mode full \
--stackoverflow_path ./data/data/train_data/raw/combined.csv \
--epochs 5 \
--output_dir ./models/design_mining
```

### 3. Transfer Learning Pipeline

For the full transfer learning approach (Sections 4.2.1-4.2.2):

#### Option A: File-based TAWOS Data

```bash
python train_design_classifier.py --mode transfer \
    --stackoverflow_path ./data/stackoverflow/data.csv \
    --tawos_path ./data/tawos/issues.csv \
    --confidence_threshold 0.8 \
    --labeled_output ./output/tawos_labeled.csv \
    --output_dir ./models/transfer_learned
```

#### Option B: TAWOS MySQL Database Connection

Connect directly to a TAWOS MySQL database to fetch issues:

```bash
# Credentials default from TAWOS_DB_* environment variables (see .environment_variables)
python train_design_classifier.py --mode transfer \
    --stackoverflow_path ./data/stackoverflow/data.csv \
    --tawos_projects MESOS FAB SERVER \
    --include_comments \
    --max_tawos_issues 10000 \
    --confidence_threshold 0.8 \
    --labeled_output ./output/tawos_labeled.csv \
    --output_dir ./models/transfer_learned
```

You can also pass credentials explicitly via `--db_host`, `--db_port`, `--db_name`, `--db_user`, `--db_password`.

#### Test TAWOS Database Connection

```bash
# Credentials default from TAWOS_DB_* environment variables
# List available projects
python tawos_connector.py --list-projects

# Show database schema
python tawos_connector.py --schema

# Fetch sample issues
python tawos_connector.py --sample 10
```

## Training Pipeline

### Stage 1: Fine-tune on Stack Overflow
Fine-tune BERT on the pre-labeled Stack Overflow dataset until F1 ≥ 0.90.

### Stage 2: Label TAWOS Dataset
Use the fine-tuned model to generate labels for TAWOS tickets with confidence scores.

### Stage 3: Second Fine-tuning
Fine-tune again on high-confidence TAWOS samples with:
- Smaller learning rate (1e-5)
- Fewer epochs (2)

## Evaluation Metrics

As defined in dissertation Section 4.2.5:

| Metric | Formula | Description |
|--------|---------|-------------|
| Accuracy | (TP + TN) / Total | Overall correctness |
| Precision | TP / (TP + FP) | Design prediction accuracy |
| Recall | TP / (TP + FN) | Design detection rate |
| F1 Score | 2 × (P × R) / (P + R) | Harmonic mean |
| AUC | (TPR + TNR) / 2 | Area under curve |

## Project Structure

```
design-detection-and-defect-improvement/
├── train_design_classifier.py  # Main training script
├── tawos_connector.py          # TAWOS MySQL database connector
├── generate_analysis_report.py # Visualization and analysis generator
├── analyze_metrics.py          # Metrics analysis utilities
├── hyperparameter_search.py    # Grid search for hyperparameters
├── software_stopwords.txt      # Domain-specific stopwords
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── TUNING_GUIDE.md             # Hyperparameter tuning guide
├── TRAINING_REPORT.md          # Comprehensive training results
├── data/                       # Data directory
│   ├── stackoverflow/
│   └── tawos/
├── models/                     # Trained models
│   └── design_mining/
├── analysis_output/            # Generated visualizations
│   ├── *.png                   # Charts and plots
│   └── *.csv                   # Statistical summaries
└── model_output/               # Default model output directory
```

## Usage Examples

### Command-Line Interface

#### Basic Training

```bash
# Demo mode with sample data
python train_design_classifier.py --mode demo --epochs 3

# Full training with real data
python train_design_classifier.py --mode full \
    --stackoverflow_path ./data/train.csv \
    --epochs 7 \
    --learning_rate 5e-5 \
    --batch_size 16 \
    --dropout 0.1 \
    --output_dir ./models/bert_optimized
```

#### Training with Validation Split

```bash
# Use separate validation dataset
python train_design_classifier.py --mode full \
    --stackoverflow_path ./data/train.csv \
    --val_data ./data/validation.csv \
    --epochs 10 \
    --output_dir ./models/with_validation
```

#### Using Different Model Architectures

The trainer supports multiple transformer architectures via the `Config.MODEL_NAME` setting:

```bash
# Edit train_design_classifier.py Config class, or use Python API:
# - bert-base-uncased (default, best performance)
# - distilbert-base-uncased (faster, competitive)
# - roberta-base (alternative tokenization)
```

#### Transfer Learning with TAWOS Database

```bash
# Full pipeline: StackOverflow → TAWOS labeling → Fine-tuning
# DB credentials default from TAWOS_DB_* environment variables
python train_design_classifier.py --mode transfer \
    --stackoverflow_path ./data/stackoverflow.csv \
    --tawos_projects MESOS FAB SERVER \
    --include_comments \
    --confidence_threshold 0.8 \
    --labeled_output ./output/labeled_tawos.csv \
    --output_dir ./models/transfer
```

### Python API

#### Training with Custom Parameters

```python
from train_design_classifier import Config, DesignMiningTrainer

# Customize configuration
config = Config()
config.MODEL_NAME = 'bert-base-uncased'
config.BATCH_SIZE = 16
config.NUM_EPOCHS = 7
config.LEARNING_RATE = 5e-5
config.DROPOUT_RATE = 0.1

# Initialize trainer
trainer = DesignMiningTrainer(config)

# Create data loaders and train
train_loader, val_loader, test_loader = trainer.create_data_loaders(
    train_texts, train_labels,
    val_texts, val_labels,
    test_texts, test_labels
)
history = trainer.train(train_loader, val_loader)

# Evaluate on test set
test_loss, test_metrics = trainer.evaluate(test_loader)
print(f"Test F1: {test_metrics['f1_score']:.4f}")
```

#### Making Predictions

```python
from train_design_classifier import DesignMiningTrainer, Config

# Load trained model
config = Config()
trainer = DesignMiningTrainer(config)
trainer.load_model('./models/bert_optimized')

# Predict
texts = [
    "Design the authentication microservice architecture",
    "Fix null pointer exception in user login"
]
predictions, confidences = trainer.predict(texts)

for text, pred, conf in zip(texts, predictions, confidences):
    label = "design" if pred == 1 else "general"
    print(f"{label} ({conf:.2f}): {text[:50]}...")
```

#### Connecting to TAWOS Database

```python
from tawos_connector import TAWOSConnector, TAWOSConfig

# Configure connection
config = TAWOSConfig(
    host="localhost",
    port=3306,
    database="tawos",
    user="root",
    password="password",
    projects=["MESOS", "FAB"],
    max_issues=5000
)

# Connect and fetch issues
connector = TAWOSConnector(config)
if connector.connect():
    # List available projects
    projects = connector.list_projects()
    for p in projects:
        print(f"{p['project_key']}: {p['issue_count']} issues")

    # Fetch issues with comments
    df = connector.fetch_issues(include_comments=True)
    print(f"Fetched {len(df)} issues")

    # Access text for classification
    texts = df['text'].tolist()

    connector.disconnect()
```

#### Full Transfer Learning Pipeline

```python
from train_design_classifier import Config, TransferLearningPipeline
from tawos_connector import TAWOSConfig

# Configure models
config = Config()
config.MODEL_NAME = 'bert-base-uncased'
config.NUM_EPOCHS = 5

# Configure TAWOS connection
tawos_config = TAWOSConfig(
    host="localhost",
    database="tawos",
    user="root",
    password="password",
    projects=["MESOS", "FAB"],
    max_issues=10000
)

# Run pipeline
pipeline = TransferLearningPipeline(config, tawos_config=tawos_config)

results = pipeline.run_full_pipeline_with_db(
    so_texts=stackoverflow_texts,
    so_labels=stackoverflow_labels,
    tawos_projects=["MESOS", "FAB"],
    include_comments=True,
    confidence_threshold=0.8,
    output_path="./output/labeled_tawos.csv"
)

print(f"Labeled {results['tawos_fetched']} TAWOS issues")
print(f"Design issues: {results['stage2']['design_predicted']}")

# Save final model
pipeline.trainer.save_model('./models/transfer_final')
```

## CLI Arguments Reference

```
python train_design_classifier.py --help
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `demo` | Training mode: `demo`, `full`, or `transfer` |
| `--stackoverflow_path` | - | Path to Stack Overflow training CSV |
| `--tawos_path` | - | Path to TAWOS CSV (file-based mode) |
| `--val_data` | - | Path to separate validation CSV |
| `--output_dir` | `./model_output` | Directory to save trained model |
| `--epochs` | `5` | Number of training epochs |
| `--learning_rate` | `2e-5` | Learning rate |
| `--batch_size` | `16` | Training batch size |
| `--dropout` | `0.1` | Dropout rate (0.0-1.0) |
| `--max_length` | `512` | Maximum sequence length |
| `--min_words` | `7` | Minimum meaningful words after stopword removal |
| `--warmup_ratio` | `0.1` | Learning rate warmup ratio |

### TAWOS Database Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--db_host` | `localhost` | MySQL database host |
| `--db_port` | `3306` | MySQL database port |
| `--db_name` | `tawos` | Database name |
| `--db_user` | `root` | Database username |
| `--db_password` | - | Database password |
| `--tawos_projects` | - | List of Apache project keys (e.g., `HADOOP SPARK`) |
| `--include_comments` | `false` | Include JIRA comment text |
| `--max_tawos_issues` | - | Maximum issues to fetch |
| `--confidence_threshold` | `0.8` | Label confidence threshold |
| `--labeled_output` | - | Path to save labeled TAWOS CSV |

## Baseline Comparison

The system compares BERT against traditional methods as in Mahadi et al.:

```bash
python train_design_classifier.py --mode demo
```

Expected output:
```
Model Comparison Summary
============================================================
Model                     Accuracy     F1 Score     AUC
------------------------------------------------------------
BERT                      0.9200       0.8750       0.9100
logistic_regression       0.8500       0.7800       0.8200
svm                       0.8600       0.7900       0.8400
```

## Research Questions Addressed

### RQ1: Design Mining with Transformers ✓
- BERT fine-tuning for design classification
- Comparison with state-of-the-art

### RQ2: Predicting Defect Rate Changes (Future Work)
- Use classified design issues to predict defect correlation

### RQ3: Cross-Project Generalization (Future Work)
- Test model performance across different projects

## References

Key papers from the dissertation:

1. Devlin et al. (2019) - BERT: Pre-training of Deep Bidirectional Transformers
2. Mahadi, Ernst, & Tongay (2021) - Conclusion stability for natural language based mining
3. Tawosi et al. (2022) - Deep Learning for Agile Effort Estimation

## License

This implementation is for academic purposes based on publicly available research.

## Citation

If using this code for research, please cite:

```bibtex
@phdthesis{morgan2026algorithmic,
  title={An Algorithmic Approach to Understanding the Impact of Design Work in Software Projects},
  author={Morgan, Steven},
  year={2026},
  school={Binghamton University}
}
```
