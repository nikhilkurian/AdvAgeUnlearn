# Adversarial Debiasing for Medical Image Classification

This repository implements a sophisticated PyTorch Lightning pipeline for training multilabel chest X-ray classifiers (14 conditions) with **adversarial debiasing** to remove age-related bias from learned features. The system uses Gradient Reversal Layer (GRL) and scheduled lambda training to achieve fair, age-invariant representations.

## 🎯 Key Features

- **Multi-label Disease Classification**: 14 chest X-ray conditions
- **Adversarial Age Debiasing**: Removes age bias using GRL
- **Scheduled Lambda Training**: Ganin-style progressive adversarial strength
- **Flexible Architecture**: Support for categorical and ordinal age prediction
- **Comprehensive Monitoring**: Real-time bias evaluation and metrics
- **Linear Probe Analysis**: Post-training fairness evaluation
- **Robust Error Handling**: Graceful handling of missing files and data issues

## 🔬 Technical Innovations

### **1. Advanced Adversarial Training**
- **Gradient Reversal Layer (GRL)**: Reverses gradients to remove age features
- **Scheduled Lambda**: Progressive adversarial strength (0 → λ_max)
- **Separate Age Adversary**: Configurable capacity with dropout
- **Correct Loss Function**: `L_total = L_main + λ * L_age` (with GRL gradient flipping)

### **2. Flexible Age Prediction Modes**
- **Categorical Mode**: Standard 4-way age group classification
- **Ordinal Mode**: Cumulative probability prediction (P(age > X))
- **Configurable Architecture**: Hidden layers, dropout, capacity

### **3. Robust Training Pipeline**
- **Gradient Clipping**: Prevents training instability
- **Early Stopping**: Prevents overfitting
- **Comprehensive Logging**: All metrics tracked in TensorBoard
- **Checkpoint Management**: Automatic best model saving

## 🚨 Recent Critical Fixes (v2.0)

### **Fix 1: Correct Loss Sign**
- **Issue**: Previous implementation used `L_main - λ * L_age` which caused training collapse
- **Solution**: Changed to `L_main + λ * L_age` with GRL gradient flipping
- **Impact**: Proper adversarial dynamics between feature extractor and age adversary

### **Fix 2: Always Use GRL**
- **Issue**: Used `feat.detach()` when λ=0, preventing age adversary warm-up
- **Solution**: Always pass through GRL, even when λ=0
- **Impact**: Age adversary can learn properly during warm-up phase

### **Performance Impact**
These fixes significantly improve adversarial debiasing effectiveness:
- **Better age bias removal** (expected linear probe AUC < 0.6)
- **Maintained main task performance** (stable disease classification)
- **More stable training dynamics** (reduced collapse risk)

## 1) Requirements

- Python 3.8+
- CUDA-capable GPU recommended (for adversarial training)
- Install Python packages:

```bash
pip install -r requirements.txt
```

### **Key Dependencies:**
- **PyTorch Lightning**: Training framework
- **TorchVision**: Pre-trained models (DenseNet121, ResNet34, ViT)
- **TensorBoard**: Training monitoring
- **scikit-learn**: Metrics and preprocessing
- **pandas**: Data handling and CSV operations

## 2) Data Expectations

You need two kinds of CSV files:

### **1) Dataset CSVs (for train/val/test)**, e.g. `train_mimic.csv`:
- **Required column**: `path_preproc` (relative path to preprocessed image)
- **Required columns**: 14 label columns with 0/1 values:
  - No Finding, Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Lung Lesion, Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax, Pleural Effusion, Pleural Other, Fracture, Support Devices
- **Optional columns**: `age` or `AgeGroup` (if `age` exists, age groups can be inferred)

### **2) Metadata CSVs (for train/val/test)**, e.g. `train_final.csv`:
- **Required column**: `img_path` (absolute path to the image)
- **Required column**: `age` (numeric age for adversarial training)
- **Optional columns**: StudyDate, Sex, View, etc. for merging into outputs

**Note on paths**: If dataset images and metadata `img_path` use different root prefixes, the training script automatically remaps dataset paths to match metadata before merging predictions/embeddings.

## 3) Architecture & Training Details

### **🏗️ Model Architecture**

#### **DenseNetAgeAdv (Primary Model)**
- **Backbone**: DenseNet121 (pre-trained on ImageNet)
- **Feature Extraction**: Global Average Pooling (1024-dim features)
- **Main Head**: Linear layer (1024 → 14) for disease classification
- **Age Adversary**: Separate MLP with configurable capacity
  - Hidden layer: 1024 → 256 → 4 (categorical) or 3 (ordinal)
  - Dropout: 0.2 (configurable)
  - Activation: ReLU

#### **Other Models**
- **DenseNet**: Standard DenseNet121 with disease classification
- **ResNet**: ResNet34 with disease classification  
- **VisionTransformer**: ViT with disease classification

### **⚡ Adversarial Training Process**

#### **1. Forward Pass**
```python
# Extract features from backbone
features = backbone(images)  # [B, 1024]

# Main task: disease classification
disease_logits = cls_head(features)  # [B, 14]
main_loss = BCEWithLogitsLoss(disease_logits, disease_labels)

# Adversarial task: age prediction with GRL
age_features = GRL(features, lambda)  # Gradient reversal
age_logits = age_adversary(age_features)  # [B, 4] or [B, 3]
age_loss = CrossEntropyLoss(age_logits, age_targets)  # or ordinal_loss
```

#### **2. Loss Calculation**
```python
# CORRECT adversarial loss with GRL
total_loss = main_loss + lambda * age_loss

# GRL flips gradients for features, creating adversarial dynamics:
# - Feature extractor tries to maximize age loss (via GRL)
# - Age adversary tries to minimize age loss  
# - Main task loss is unaffected by GRL
# - Always run through GRL (even when λ=0) for proper warm-up
```

#### **3. Lambda Scheduling**
```python
# Ganin-style progressive scheduling
lambda = lambda_max * (2.0 / (1.0 + exp(-10 * progress)) - 1.0)

# Starts at ~0, gradually increases to lambda_max
# Prevents training instability from high lambda at start
```

### **🎯 Age Prediction Modes**

#### **Categorical Mode (Default)**
- **Output**: 4 logits [P(0-36), P(36-50), P(50-65), P(65+)]
- **Loss**: CrossEntropyLoss
- **Prediction**: argmax(logits)

#### **Ordinal Mode**
- **Output**: 3 logits [P(age>36), P(age>50), P(age>65)]
- **Loss**: BinaryCrossEntropy (cumulative probabilities)
- **Prediction**: Use cumulative logic for age group

### **📊 Training Monitoring**

#### **Logged Metrics**
- **Main Task**: Per-label AUROC, macro/micro AUROC
- **Adversarial**: Age prediction loss, current lambda value
- **Training**: Total loss, learning rate, gradient norms
- **Visualization**: Sample training images

#### **Real-time Bias Evaluation**
- Lambda progression over training steps
- Age prediction accuracy (should decrease)
- Main task performance (should maintain)

## 4) Configure

All settings are in `config.yaml`. The file includes comprehensive documentation for all parameters.

### **Essential Configuration:**

```yaml
# Model Architecture
model_type: 'DenseNetAgeAdv'  # Options: DenseNet, ResNet, VisionTransformer, DenseNetAgeAdv
num_classes_main: 14

# Adversarial Training Settings
debias_enable: true                    # Master switch for debiasing
adv_age_lambda: 0.5                    # Maximum lambda value
use_scheduled_lambda: true             # Ganin-style scheduling
age_mode: "categorical"                # "categorical" or "ordinal"
age_head_hidden: 256                   # Adversary capacity
age_head_dropout: 0.2                  # Adversary dropout
lambda_warmup_frac: 0.3                # Warm-up fraction

# Data paths (update these)
train_csv: '/path/to/train_mimic.csv'
val_csv: '/path/to/val_mimic.csv'
test_csv: '/path/to/test_mimic.csv'
train_details_csv: '/path/to/train_final.csv'
val_details_csv: '/path/to/val_final.csv'
test_details_csv: '/path/to/test_final.csv'
img_data_dir: '/path/to/images_root/'

# Training parameters
batch_size_main: 64                    # Reduced from 150 to prevent OOM
epochs_main: 50
enable_early_stopping: True
early_stopping_patience: 10

# Age groups for adversarial training
age_groups:
  - [0, 36, "0-36"]
  - [36, 50, "36-50"]
  - [50, 65, "50-65"]
  - [65, -1, "65+"]
```

### **Configuration Examples:**

```yaml
# Conservative settings (start here)
adv_age_lambda: 0.3
age_head_hidden: 128
age_head_dropout: 0.3

# Aggressive settings (stronger debiasing)
adv_age_lambda: 0.8
age_head_hidden: 512
age_head_dropout: 0.1

# Baseline (no debiasing)
debias_enable: false

# Ordinal mode
age_mode: "ordinal"
```

## 5) Run Training

### **Basic Training:**
```bash
python train.py --config config.yaml --dev 0
```

### **Resume from Checkpoint:**
```bash
python train.py --config config.yaml --checkpoint_path /path/to/checkpoint.ckpt --dev 0
```

### **Monitor Training:**
```bash
tensorboard --logdir logs/
```

### **Key Training Features:**
- **Automatic Lambda Scheduling**: Lambda progresses from 0 to max value
- **Gradient Clipping**: Prevents training instability
- **Early Stopping**: Stops if validation loss doesn't improve
- **Comprehensive Logging**: All metrics tracked in real-time
- **Checkpoint Management**: Saves best model automatically

### **Expected Training Behavior:**
1. **Lambda Progression**: Should start near 0, gradually increase to max value
2. **Age Loss**: Should increase over time (harder to predict age)
3. **Main Task Loss**: Should remain stable or improve slightly
4. **Validation Metrics**: Should maintain or improve performance

## 6) Evaluate Fairness (Linear Probe)

After training, evaluate how well age bias was removed:

### **Run Linear Probe:**
```bash
python linear_probe_age.py --config linear_probe_config.yaml
```

### **What Linear Probe Does:**
1. **Loads trained model** from checkpoint
2. **Extracts embeddings** from feature layer
3. **Trains linear classifier** to predict age from embeddings
4. **Reports AUC scores** - lower scores = better debiasing

### **Interpret Results:**
- **AUC < 0.6**: Excellent debiasing (age features removed)
- **AUC 0.6-0.7**: Good debiasing (moderate age bias)
- **AUC > 0.7**: Poor debiasing (strong age bias remains)

### **Monitor Probe Training:**
```bash
tensorboard --logdir outputs/linear_probe_*/logs/
```

## 7) Outputs

### **Training Outputs:**
For each run, files are written under the output directory specified in `config.yaml`:

#### **Predictions & Embeddings:**
- `predictions.train.epoch_X.csv` - Training predictions with metadata
- `predictions.val.model_<id>.csv` - Validation predictions with metadata
- `predictions.test.model_<id>.csv` - Test predictions with metadata
- `embeddings.train.epoch_X.csv` - Training embeddings (1024-dim features)
- `embeddings.val.model_<id>.csv` - Validation embeddings
- `embeddings.test.model_<id>.csv` - Test embeddings

#### **Metrics & Checkpoints:**
- `per_class_auc.txt` - Per-class and macro AUROC scores
- `checkpoints/` - Best and last model checkpoints
- `logs/` - TensorBoard logs for monitoring

### **Linear Probe Outputs:**
- `outputs/linear_probe_*/age_probe_results.txt` - Fairness evaluation results
- `outputs/linear_probe_*/embeddings/` - Extracted features for analysis
- `outputs/linear_probe_*/logs/` - Probe training logs

### **File Contents:**
- **Predictions CSV**: Per-class probabilities, logits, targets, `img_path`, merged metadata
- **Embeddings CSV**: 1024-dim feature vectors, targets, metadata
- **Results TXT**: AUC scores, class distribution, fairness metrics

## 8) Troubleshooting

### **Common Issues & Solutions:**

#### **Training Issues:**
- **Empty CSVs after run**: Ensure `img_path` in metadata matches dataset image paths. The script automatically remaps common roots.
- **CUDA out of memory**: Reduce `batch_size_main` or `image_size`
- **Slow data loading**: Increase `num_workers` (but not beyond CPU cores)
- **File not found**: Verify all paths in `config.yaml` are absolute and readable

#### **Adversarial Training Issues:**
- **High age prediction accuracy**: Increase `adv_age_lambda` or `age_head_hidden`
- **Training instability**: Reduce `adv_age_lambda`, increase `lambda_warmup_frac`
- **Poor main task performance**: Reduce `adv_age_lambda`, check `debias_enable`
- **Lambda not changing**: Verify `use_scheduled_lambda: true`

#### **Linear Probe Issues:**
- **High AUC scores**: Model still contains age bias - retrain with higher lambda
- **Checkpoint not found**: Verify checkpoint path in `linear_probe_config.yaml`
- **Empty probe results**: Check if embeddings were extracted correctly

### **Performance Tuning:**

#### **For Better Debiasing:**
```yaml
adv_age_lambda: 0.8          # Higher lambda
age_head_hidden: 512         # More capacity
age_head_dropout: 0.1        # Less regularization
lambda_warmup_frac: 0.2      # Faster ramp-up
```

#### **For Training Stability:**
```yaml
adv_age_lambda: 0.3          # Lower lambda
age_head_hidden: 128         # Less capacity
age_head_dropout: 0.3        # More regularization
lambda_warmup_frac: 0.4      # Slower ramp-up
```

### **Debugging Tips:**
- **Monitor lambda progression** in TensorBoard
- **Check age prediction loss** - should increase over time
- **Verify main task performance** - should remain stable
- **Use linear probe** to quantitatively measure bias removal

## 9) Performance Expectations

### **After Recent Fixes (v2.0):**

#### **Training Performance:**
- **Lambda Schedule**: Should progress smoothly from ~0.005 to 0.5
- **Age Loss**: Should increase over time (indicating successful debiasing)
- **Main Task AUC**: Should maintain >0.8 macro AUC
- **Training Stability**: No collapse or divergence

#### **Fairness Results:**
- **Linear Probe AUC**: Target <0.6 (excellent debiasing)
- **Age Prediction Accuracy**: Should decrease over training
- **Cross-Age Performance**: Similar performance across age groups

#### **Expected Timeline:**
- **Epochs 1-5**: Lambda warm-up, age adversary learning
- **Epochs 5-20**: Active debiasing, age loss increasing
- **Epochs 20+**: Stabilization, fine-tuning

### **Baseline Comparison:**
- **Without Debiasing**: Linear probe AUC ~0.7-0.8
- **With Debiasing**: Linear probe AUC ~0.5-0.6
- **Main Task**: Minimal performance drop (<2% AUC)

## 10) Advanced Usage

### **Custom Lambda Schedules:**
```python
# Modify schedule_lambda function in model.py
def custom_schedule(step, total_steps, lambda_max):
    # Your custom scheduling logic
    return lambda_max * (step / total_steps)  # Linear schedule
```

### **Multi-GPU Training:**
```bash
python train.py --config config.yaml --dev 0,1,2,3
```

### **Hyperparameter Search:**
```bash
# Test different lambda values
for lambda in 0.3 0.5 0.8; do
    sed -i "s/adv_age_lambda: .*/adv_age_lambda: $lambda/" config.yaml
    python train.py --config config.yaml --dev 0
done
```

### **Ensemble Methods:**
```bash
# Train multiple models with different seeds
for seed in 42 123 456; do
    python train.py --config config.yaml --seed $seed --dev 0
done
``` 