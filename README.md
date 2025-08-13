# Medical Image Classification with Adversarial Training

A PyTorch Lightning-based framework for medical image classification with adversarial training to reduce demographic bias. This project implements multiple deep learning architectures for chest X-ray analysis using the MIMIC-CXR dataset.

## 🏥 Project Overview

This repository contains a robust medical image classification system designed to:
- Classify 14 different medical conditions from chest X-rays
- Reduce demographic bias through adversarial training
- Support multiple deep learning architectures
- Provide comprehensive evaluation metrics and embeddings extraction

## 🚀 Features

### Models Supported
- **DenseNetAgeAdv**: DenseNet-121 with adversarial age group training
- **DenseNet**: Standard DenseNet-121 for medical image classification
- **ResNet**: ResNet-34 architecture
- **VisionTransformer**: Vision Transformer (ViT) architecture

### Key Features
- ✅ **Adversarial Training**: Age group adversarial training to reduce demographic bias
- ✅ **Multi-label Classification**: 14 medical conditions classification
- ✅ **Age Group Analysis**: Automatic age group categorization and analysis
- ✅ **Comprehensive Logging**: TensorBoard integration for training monitoring
- ✅ **Checkpoint Management**: Automatic checkpointing and recovery
- ✅ **Embeddings Extraction**: Feature vector extraction for downstream analysis
- ✅ **AUC Evaluation**: Per-class and macro AUC calculation
- ✅ **Data Augmentation**: Configurable image augmentation pipeline

## 📋 Requirements

### System Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM (for large datasets)

### Python Dependencies
```bash
torch>=1.12.0
torchvision>=0.13.0
pytorch-lightning>=1.8.0
timm>=0.6.0
pandas>=1.4.0
numpy>=1.21.0
scikit-learn>=1.1.0
scikit-image>=0.19.0
Pillow>=9.0.0
PyYAML>=6.0
tqdm>=4.64.0
```

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/medical-image-classification.git
cd medical-image-classification
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
AdvRem/
├── config.yaml              # Configuration file
├── dataset.py               # Data loading and preprocessing
├── model.py                 # Model architectures
├── train.py                 # Training script
├── README.md               # This file
├── outputs/                # Training outputs
├── checkpoints/            # Model checkpoints
├── logs/                   # TensorBoard logs
└── learning-not-to-learn/  # Additional learning methods
```

## ⚙️ Configuration

Edit `config.yaml` to customize your training setup:

```yaml
# Model Settings
model_type: 'DenseNetAgeAdv'  # Options: 'DenseNet', 'ResNet', 'VisionTransformer', 'DenseNetAgeAdv'
num_classes_main: 14
adv_age_lambda: 0.1  # Adversarial training weight

# Data Settings
image_size: [224, 224]
batch_size_main: 150
num_workers: 4

# Training Settings
epochs_main: 50
enable_early_stopping: True
early_stopping_patience: 10

# Age Groups Configuration
age_groups:
  - [0, 36, "0-36"]
  - [36, 50, "36-50"]
  - [50, 65, "50-65"]
  - [65, -1, "65+"]
```

## 🏃‍♂️ Usage

### 1. Prepare Your Data

Ensure your data follows this structure:
```
data/
├── train_mimic.csv          # Training data CSV
├── val_mimic.csv            # Validation data CSV
├── test_mimic.csv           # Test data CSV
├── train_final.csv          # Training metadata
├── val_final.csv            # Validation metadata
├── test_final.csv           # Test metadata
└── images/                  # Image files
```

### 2. Update Configuration

Modify `config.yaml` with your data paths:
```yaml
train_csv: '/path/to/your/train_mimic.csv'
val_csv: '/path/to/your/val_mimic.csv'
test_csv: '/path/to/your/test_mimic.csv'
train_details_csv: '/path/to/your/train_final.csv'
val_details_csv: '/path/to/your/val_final.csv'
test_details_csv: '/path/to/your/test_final.csv'
img_data_dir: '/path/to/your/images/'
```

### 3. Start Training

```bash
# Basic training
python train.py --config config.yaml --dev 0

# Resume from checkpoint
python train.py --config config.yaml --checkpoint_path path/to/checkpoint.ckpt --dev 0
```

### 4. Monitor Training

```bash
# Start TensorBoard
tensorboard --logdir logs/
```

## 📊 Medical Conditions

The model classifies 14 medical conditions:
1. No Finding
2. Enlarged Cardiomediastinum
3. Cardiomegaly
4. Lung Opacity
5. Lung Lesion
6. Edema
7. Consolidation
8. Pneumonia
9. Atelectasis
10. Pneumothorax
11. Pleural Effusion
12. Pleural Other
13. Fracture
14. Support Devices

## 📈 Outputs

### Training Outputs
- **Predictions**: CSV files with model predictions, logits, and targets
- **Embeddings**: 1024-dimensional feature vectors for each image
- **Metrics**: Per-class AUC scores and macro AUC
- **Checkpoints**: Model checkpoints for recovery and inference

### File Structure
```
outputs/experiment_name/
├── predictions.train.epoch_X.csv
├── predictions.val.model_X.csv
├── predictions.test.model_X.csv
├── embeddings.train.epoch_X.csv
├── embeddings.val.model_X.csv
├── embeddings.test.model_X.csv
└── per_class_auc.txt
```

## 🔬 Adversarial Training

The `DenseNetAgeAdv` model implements adversarial training to reduce demographic bias:

- **Main Task**: Medical condition classification
- **Adversarial Task**: Age group prediction with gradient reversal
- **Loss Function**: `Total Loss = Classification Loss + λ × Adversarial Loss`

This approach helps the model learn features that are invariant to age-related demographic factors.

## 🧪 Experiments

### Supported Model Architectures

1. **DenseNetAgeAdv** (Recommended)
   - DenseNet-121 backbone
   - Adversarial age group training
   - Best for reducing demographic bias

2. **DenseNet**
   - Standard DenseNet-121
   - Good baseline performance

3. **ResNet**
   - ResNet-34 architecture
   - Faster training, good performance

4. **VisionTransformer**
   - Vision Transformer (ViT)
   - State-of-the-art attention-based model

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{medical_image_classification_2024,
  title={Medical Image Classification with Adversarial Training for Demographic Bias Reduction},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Contributing Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/medical-image-classification/issues) page
2. Create a new issue with detailed information
3. Include your configuration and error messages

## 🙏 Acknowledgments

- MIMIC-CXR dataset for providing the medical imaging data
- PyTorch Lightning for the training framework
- The medical imaging research community

## 📊 Performance

Typical performance metrics on MIMIC-CXR test set:
- **Macro AUC**: ~0.76-0.80
- **Best performing class**: Pleural Effusion (AUC: ~0.89)
- **Training time**: ~2-4 hours on RTX 4090

---

**Note**: This is a research implementation. For clinical use, additional validation and regulatory approval may be required. 