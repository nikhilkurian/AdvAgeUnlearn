# Linear Probe for Age Group Analysis

This script evaluates how well age group information can be predicted from embeddings learned by the DenseNetAgeAdv model.

## What it does

1. **Loads the trained DenseNetAgeAdv model** from the specified checkpoint
2. **Extracts 1024-dimensional embeddings** from the model's feature layer
3. **Trains a simple linear classifier** to predict age groups from these embeddings
4. **Evaluates performance** using per-class and macro AUC scores

## Purpose

This is a **fairness evaluation tool**. It measures whether the learned embeddings contain age-related information:

- **High AUC scores** = embeddings contain strong age signals (potential bias)
- **Low AUC scores** = embeddings are age-invariant (good fairness)

## Usage

1. **Update the configuration** in `linear_probe_config.yaml`:
   - Set the correct checkpoint path
   - Adjust batch size, learning rate, epochs as needed

2. **Run the analysis**:
   ```bash
   python linear_probe_age.py --config linear_probe_config.yaml
   ```

3. **Monitor training**:
   ```bash
   # The log directory will be printed when you run the script
   tensorboard --logdir outputs/linear_probe_{experiment_name}_{checkpoint_name}/logs/
   ```

## Outputs

All outputs are automatically organized in a folder named after the checkpoint being analyzed:

```
outputs/linear_probe_{experiment_name}_{checkpoint_name}/
├── logs/                    # TensorBoard logs
├── checkpoints/            # Best linear probe model
├── embeddings/             # Extracted features and labels
│   ├── train_embeddings.npy
│   ├── val_embeddings.npy
│   ├── test_embeddings.npy
│   ├── train_age_labels.npy
│   ├── val_age_labels.npy
│   └── test_age_labels.npy
└── age_probe_results.txt   # Final AUC scores and analysis
```

**Example**: For checkpoint `mimic_mdn_experiment_model_1_20250813_173013/epoch=12-val_loss=0.37.ckpt`, 
the output folder would be: `outputs/linear_probe_mimic_mdn_experiment_model_1_20250813_173013_epoch=12-val_loss=0.37/`

- **TensorBoard logs**: Training progress in `logs/`
- **Model checkpoints**: Best linear probe model in `checkpoints/`
- **Embeddings**: Extracted features and labels in `embeddings/`
- **Results file**: `age_probe_results.txt` with final AUC scores

## Configuration

Key settings in `linear_probe_config.yaml`:

```yaml
# Path to the trained model checkpoint
checkpoint_path: "/path/to/your/checkpoint.ckpt"

# Linear probe training settings
linear_probe:
  batch_size: 64
  learning_rate: 0.001
  max_epochs: 50
  num_classes: 4  # age groups
```

## Automatic Output Naming

The script automatically generates output folder names based on the checkpoint path:

- **Input**: `/path/to/checkpoints/mimic_mdn_experiment_model_1_20250813_173013/epoch=12-val_loss=0.37.ckpt`
- **Output**: `outputs/linear_probe_mimic_mdn_experiment_model_1_20250813_173013_epoch=12-val_loss=0.37/`

This ensures each linear probe analysis is clearly associated with its source model checkpoint.

## Interpretation

- **Macro AUC < 0.6**: Good - embeddings don't contain strong age signals
- **Macro AUC 0.6-0.7**: Moderate - some age information present
- **Macro AUC > 0.7**: Concerning - strong age signals in embeddings

The goal is to achieve low AUC scores, indicating that the adversarial training successfully removed age-related features from the embeddings.
