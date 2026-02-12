# Model Tuning Guide: Improving Accuracy and AUC

Based on your current results (~87% accuracy, ~90% AUC), here are evidence-based strategies to improve performance.

## Current Performance Analysis

**Strengths:**
- High validation precision (1.0000) - model is confident when it predicts design
- Good test recall (0.9032) - catching most design cases

**Weaknesses:**
- Validation recall (0.7667) - missing ~23% of design cases in validation
- Test precision (0.8750) - some false positives

## 🎯 Recommended Adjustments (Priority Order)

### 1. **Increase Training Epochs** (Highest Impact)

Your model stopped at 5 epochs. Try training longer:

```bash
# Try 7-10 epochs
python train_design_classifier.py --mode full \
    --stackoverflow_path ./data/data/train_data/raw/combined.csv \
    --val_data ./data/data/validation_data/raw/validation.csv \
    --epochs 10 \
    --output_dir ./models/design_mining_10epochs
```

**Why:** BERT often needs 8-12 epochs to converge fully on small datasets.

**Risk:** Monitor for overfitting (val loss increasing while train loss decreases).

---

### 2. **Adjust Learning Rate** (High Impact)

Current: `2e-5`

Try these alternatives:

```bash
# Lower learning rate for more careful fine-tuning
python train_design_classifier.py --mode full \
    --stackoverflow_path ./data/data/train_data/raw/combined.csv \
    --val_data ./data/data/validation_data/raw/validation.csv \
    --epochs 10 \
    --learning_rate 1e-5 \
    --output_dir ./models/design_mining_lr1e5

# Slightly higher for faster convergence
python train_design_classifier.py --mode full \
    --stackoverflow_path ./data/data/train_data/raw/combined.csv \
    --val_data ./data/data/validation_data/raw/validation.csv \
    --epochs 7 \
    --learning_rate 3e-5 \
    --output_dir ./models/design_mining_lr3e5
```

**Why:** Lower LR = more stable but slower. Higher LR = faster but may overshoot.

**Best practice:** Start with `1e-5` for careful tuning, try `3e-5` if training is slow.

---

### 3. **Increase Batch Size** (Medium Impact)

Current: `16`

```bash
# Larger batch size for more stable gradients
python train_design_classifier.py --mode full \
    --stackoverflow_path ./data/data/train_data/raw/combined.csv \
    --val_data ./data/data/validation_data/raw/validation.csv \
    --epochs 10 \
    --batch_size 32 \
    --output_dir ./models/design_mining_bs32
```

**Why:** Larger batches provide more stable gradient estimates.

**Trade-off:** Requires more GPU memory, may converge to sharper minima.

---

### 4. **Adjust Dropout Rate** (Medium Impact)

Current: `0.1` (low)

```bash
# Higher dropout to reduce overfitting
python train_design_classifier.py --mode full \
    --stackoverflow_path ./data/data/train_data/raw/combined.csv \
    --val_data ./data/data/validation_data/raw/validation.csv \
    --epochs 10 \
    --dropout 0.3 \
    --output_dir ./models/design_mining_dropout03
```

**Why:** Higher dropout forces the model to learn more robust features.

**When to use:** If you see overfitting (big gap between train and val performance).

---

### 5. **Increase Maximum Sequence Length** (Medium Impact)

Current: `512` (BERT max)

If your texts are truncated:

```bash
# Check if texts are being truncated
python train_design_classifier.py --mode full \
    --stackoverflow_path ./data/data/train_data/raw/combined.csv \
    --val_data ./data/data/validation_data/raw/validation.csv \
    --max_length 256 \
    --epochs 10 \
    --output_dir ./models/design_mining_len256
```

**Why:** Shorter sequences train faster. If most texts < 256 tokens, this helps.

**Recommendation:** Analyze your text lengths first to choose optimal length.

---

### 6. **Reduce Minimum Word Threshold** (Low-Medium Impact)

Current: `7` meaningful words

```bash
# Include shorter texts
python train_design_classifier.py --mode full \
    --stackoverflow_path ./data/data/train_data/raw/combined.csv \
    --val_data ./data/data/validation_data/raw/validation.csv \
    --min_words 5 \
    --epochs 10 \
    --output_dir ./models/design_mining_minwords5
```

**Why:** You're currently removing texts with < 7 meaningful words. Some design discussions might be concise.

**Trade-off:** More noise in training data vs more training examples.

---

### 7. **Use Different BERT Model** (Potentially High Impact)

Try these alternatives in `Config` class:

**Option A: RoBERTa** (often better for classification)
```python
MODEL_NAME = 'roberta-base'  # Instead of 'bert-base-uncased'
```

**Option B: DistilBERT** (faster, 97% performance)
```python
MODEL_NAME = 'distilbert-base-uncased'
```

**Option C: Domain-specific BERT**
```python
MODEL_NAME = 'microsoft/codebert-base'  # For code-related text
```

---

## 🔬 Recommended Experiment Sequence

### Quick Wins (Try First):

1. **Experiment 1: More Epochs**
   ```bash
   --epochs 10
   ```

2. **Experiment 2: Lower Learning Rate**
   ```bash
   --epochs 10 --learning_rate 1e-5
   ```

3. **Experiment 3: Combination**
   ```bash
   --epochs 10 --learning_rate 1e-5 --batch_size 32 --dropout 0.2
   ```

### Advanced Tuning:

4. **Experiment 4: Higher Regularization**
   ```bash
   --epochs 10 --learning_rate 1e-5 --dropout 0.3 --batch_size 8
   ```

5. **Experiment 5: Aggressive Training**
   ```bash
   --epochs 15 --learning_rate 3e-5 --batch_size 32 --dropout 0.1
   ```

---

## 📊 How to Compare Results

After running experiments, compare using the CSV:

```python
import pandas as pd

# Read metrics
df = pd.read_csv('./models/design_mining/training_metrics.csv')

# Sort by test F1 score
df_sorted = df.sort_values('test_f1_score', ascending=False)

# Show top configurations
print(df_sorted[['run_id', 'learning_rate', 'batch_size', 'num_epochs',
                  'test_accuracy', 'test_f1_score', 'test_auc']].head(10))
```

---

## 🎯 Target Metrics

Based on literature for design vs non-design classification:

- **Good:** F1 > 0.85, AUC > 0.90 ✓ (You're here!)
- **Very Good:** F1 > 0.90, AUC > 0.93
- **Excellent:** F1 > 0.95, AUC > 0.96

---

## 🚨 Warning Signs

**Overfitting:**
- Train accuracy >> Val accuracy (gap > 10%)
- Val loss increases while train loss decreases
- **Fix:** Increase dropout, reduce epochs, add more data

**Underfitting:**
- Both train and val accuracy are low
- Loss not decreasing
- **Fix:** Increase epochs, increase learning rate, increase model capacity

**Class Imbalance:**
- High accuracy but poor recall on minority class
- **Fix:** Use class weights, oversample minority class

---

## 💡 Pro Tips

1. **Always train multiple times with same config** - Results can vary by 1-2%
2. **Monitor validation metrics during training** - Stop if overfitting
3. **Save your best model** - Based on validation F1, not just loss
4. **Track everything** - The CSV helps you find patterns

---

## 🔍 Debugging Low Performance

If metrics don't improve:

1. **Check data quality:**
   ```bash
   # Look at preprocessing logs
   grep "PREPROCESSING SUMMARY" training.log
   ```

2. **Verify labels are correct:**
   ```python
   df = pd.read_csv('your_data.csv')
   print(df['label'].value_counts())
   ```

3. **Examine misclassifications:**
   - Add code to save predictions on test set
   - Manually review false positives and false negatives

---

## 📚 Advanced Techniques (If Basic Tuning Doesn't Help)

### 1. **Learning Rate Scheduling**
Currently using linear warmup. Try cosine annealing.

### 2. **Gradient Accumulation**
Simulate larger batches:
```python
# In training loop
if step % accumulation_steps == 0:
    optimizer.step()
```

### 3. **Class Weights**
If dataset is imbalanced:
```python
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=[0,1], y=train_labels)
```

### 4. **Ensemble Methods**
Train 3-5 models with different seeds, average predictions.

### 5. **Data Augmentation**
- Backtranslation
- Synonym replacement
- Random deletion

---

## 📝 Recommended Starting Point

Based on your current setup, try this first:

```bash
python train_design_classifier.py --mode full \
    --stackoverflow_path ./data/data/train_data/raw/combined.csv \
    --val_data ./data/data/validation_data/raw/validation.csv \
    --epochs 10 \
    --learning_rate 1e-5 \
    --batch_size 32 \
    --dropout 0.2 \
    --warmup_ratio 0.15 \
    --output_dir ./models/design_mining_optimized
```

This configuration:
- ✅ More epochs for better convergence
- ✅ Lower LR for careful fine-tuning
- ✅ Larger batch for stable gradients
- ✅ Moderate dropout to prevent overfitting
- ✅ Slightly more warmup

**Expected improvement:** +2-4% on all metrics

---

## 🎓 Key Takeaways

1. **Start simple:** Change one thing at a time
2. **Use the CSV:** Compare apples-to-apples
3. **Don't overtune:** The test set is not for tuning!
4. **More data > better hyperparameters:** If possible, get more training data
5. **Domain matters:** Software engineering text may need domain-specific BERT

Good luck! 🚀
