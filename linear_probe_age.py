#!/usr/bin/env python3
"""
Linear Probe for Age Group Analysis

This script evaluates how well age group information can be predicted from 
embeddings learned by the DenseNetAgeAdv model. It extracts embeddings from
the trained model and trains a simple linear classifier to predict age groups.

Usage:
    python linear_probe_age.py --config linear_probe_config.yaml
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score
import yaml
import argparse
from tqdm import tqdm

# Import our existing modules
from dataset import CheXpertDataModule
from model import DenseNetAgeAdv


class LinearProbeModel(pl.LightningModule):
    """Simple linear classifier for age group prediction from embeddings."""
    
    def __init__(self, input_dim=1024, num_classes=4, learning_rate=0.001):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.preds_probs = {'probabilities': [], 'labels': []}
        
    def forward(self, x):
        return self.fc(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = nn.CrossEntropyLoss()(output, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = nn.CrossEntropyLoss()(output, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        prob = torch.softmax(y_hat, dim=1)
        
        # Store predictions and labels
        self.preds_probs['probabilities'].append(prob.cpu().numpy())
        self.preds_probs['labels'].append(y.cpu().numpy())
        
        return None
    
    def on_test_end(self):
        # Concatenate all batches
        all_probs = np.concatenate(self.preds_probs['probabilities'], axis=0)
        all_labels = np.concatenate(self.preds_probs['labels'], axis=0)
        
        # Compute AUC
        macro_auc = self.compute_classwise_auc(all_probs, all_labels)
        print(f'Macro AUC: {macro_auc}')
        
        # Store the result for later access
        self.test_auc = macro_auc
        
        return {'test_auc': macro_auc}
    
    def compute_classwise_auc(self, probs, labels):
        """Compute per-class and macro AUC."""
        auc_per_class = []
        auc_results = []
        
        for i in range(self.num_classes):
            # Create binary labels for class i
            binary_labels = (labels == i).astype(int)
            
            # Calculate AUC for class i
            auc = roc_auc_score(binary_labels, probs[:, i])
            auc_per_class.append(auc)
            auc_results.append((i, auc))
            print(f'Class {i} AUC: {auc:.4f}')
        
        # Calculate macro AUC
        macro_auc = np.mean(auc_per_class)
        print(f'Macro AUC: {macro_auc:.4f}')
        
        # Store per-class AUCs for later access
        self.auc_per_class = auc_per_class
        
        # Save results
        auc_results.append(("Macro AUC", macro_auc))
        
        return macro_auc
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def extract_embeddings(model, dataloader, device):
    """Extract embeddings from the trained model."""
    model.eval()
    embeddings = []
    labels = []
    age_groups = []
    
    print("Extracting embeddings...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Get data from batch
            img = batch['image'].to(device)
            lab = batch['label'].to(device)
            age_group = batch['age_group']
            
            # Extract embeddings (remove head and get features)
            feat = model.extract_gap_feat(img)  # This gives us the 1024-dim features
            embeddings.append(feat.cpu())
            labels.append(lab.cpu())
            age_groups.extend(age_group)
    
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)
    
    return embeddings.numpy(), labels.numpy(), age_groups


def prepare_age_group_labels(age_groups, age_group_config):
    """Convert age group strings to numeric labels."""
    # Create age group mapping
    age_group_mapping = {}
    for i, (min_age, max_age, group_name) in enumerate(age_group_config):
        age_group_mapping[group_name] = i
    
    # Convert to numeric labels
    numeric_labels = []
    for age_group in age_groups:
        if age_group in age_group_mapping:
            numeric_labels.append(age_group_mapping[age_group])
        else:
            # Handle unknown age groups
            numeric_labels.append(0)  # Default to first group
    
    return np.array(numeric_labels)


def generate_output_paths(checkpoint_path):
    """Generate output folder names based on checkpoint path."""
    # Extract model info from checkpoint path
    # Example: /path/to/checkpoints/mimic_mdn_experiment_model_1_20250813_173013/epoch=12-val_loss=0.37.ckpt
    
    # Get the experiment folder name
    checkpoint_dir = os.path.dirname(checkpoint_path)
    experiment_name = os.path.basename(checkpoint_dir)
    
    # Get the checkpoint filename without extension
    checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    
    # Create a descriptive output folder name
    output_folder = f"linear_probe_{experiment_name}_{checkpoint_name}"
    
    # Generate all paths
    base_output_dir = f"./outputs/{output_folder}"
    
    return {
        'output_dir': base_output_dir,
        'log_dir': f"{base_output_dir}/logs",
        'checkpoints_dir': f"{base_output_dir}/checkpoints", 
        'embeddings_dir': f"{base_output_dir}/embeddings"
    }

def main(args):
    # Load configuration
    with open(args.config, 'r') as f:
        probe_config = yaml.safe_load(f)
    
    # Load main config for data paths
    with open(probe_config['main_config_path'], 'r') as f:
        main_config = yaml.safe_load(f)
    
    # Generate output paths based on checkpoint
    output_paths = generate_output_paths(probe_config['checkpoint_path'])
    
    # Update config with generated paths
    probe_config.update(output_paths)
    
    print(f"Output directory: {probe_config['output_dir']}")
    print(f"Log directory: {probe_config['log_dir']}")
    print(f"Checkpoints directory: {probe_config['checkpoints_dir']}")
    print(f"Embeddings directory: {probe_config['embeddings_dir']}")
    
    # Create output directories
    os.makedirs(probe_config['output_dir'], exist_ok=True)
    os.makedirs(probe_config['log_dir'], exist_ok=True)
    os.makedirs(probe_config['checkpoints_dir'], exist_ok=True)
    os.makedirs(probe_config['embeddings_dir'], exist_ok=True)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the trained DenseNetAgeAdv model
    print(f"Loading model from: {probe_config['checkpoint_path']}")
    model = DenseNetAgeAdv.load_from_checkpoint(
        probe_config['checkpoint_path'],
        num_classes=main_config['num_classes_main'],
        num_age_groups=4,
        adv_age_lambda=main_config.get('adv_age_lambda', 0.5),
        use_scheduled_lambda=main_config.get('use_scheduled_lambda', True),
        age_mode=main_config.get('age_mode', 'categorical'),
        age_head_hidden=main_config.get('age_head_hidden', 256),
        age_head_dropout=main_config.get('age_head_dropout', 0.2),
        debias_enable=main_config.get('debias_enable', True)
    )
    model.to(device)
    
    # Remove the classification head to get embeddings
    model.remove_head()
    
    # Setup data module
    print("Setting up data...")
    age_groups_config = []
    for min_age, max_age, group_name in main_config['age_groups']:
        if max_age == -1:
            age_groups_config.append((min_age, float('inf'), group_name))
        else:
            age_groups_config.append((min_age, max_age, group_name))
    
    data_module = CheXpertDataModule(
        csv_train_img=main_config['train_csv'],
        csv_val_img=main_config['val_csv'],
        csv_test_img=main_config['test_csv'],
        image_size=tuple(main_config['image_size']),
        img_data_dir=main_config['img_data_dir'],
        pseudo_rgb=True,
        batch_size=probe_config['linear_probe']['batch_size'],
        num_workers=main_config['num_workers'],
        age_groups=age_groups_config
    )
    
    # Extract embeddings for all splits
    print("Extracting embeddings from train set...")
    train_embeddings, train_labels, train_age_groups = extract_embeddings(
        model, data_module.train_dataloader(), device
    )
    
    print("Extracting embeddings from validation set...")
    val_embeddings, val_labels, val_age_groups = extract_embeddings(
        model, data_module.val_dataloader(), device
    )
    
    print("Extracting embeddings from test set...")
    test_embeddings, test_labels, test_age_groups = extract_embeddings(
        model, data_module.test_dataloader(), device
    )
    
    # Prepare age group labels
    print("Preparing age group labels...")
    train_age_labels = prepare_age_group_labels(train_age_groups, age_groups_config)
    val_age_labels = prepare_age_group_labels(val_age_groups, age_groups_config)
    test_age_labels = prepare_age_group_labels(test_age_groups, age_groups_config)
    
    # Save embeddings for future analysis
    print("Saving embeddings...")
    np.save(os.path.join(probe_config['embeddings_dir'], 'train_embeddings.npy'), train_embeddings)
    np.save(os.path.join(probe_config['embeddings_dir'], 'val_embeddings.npy'), val_embeddings)
    np.save(os.path.join(probe_config['embeddings_dir'], 'test_embeddings.npy'), test_embeddings)
    np.save(os.path.join(probe_config['embeddings_dir'], 'train_age_labels.npy'), train_age_labels)
    np.save(os.path.join(probe_config['embeddings_dir'], 'val_age_labels.npy'), val_age_labels)
    np.save(os.path.join(probe_config['embeddings_dir'], 'test_age_labels.npy'), test_age_labels)
    
    # Compute class weights for balanced training
    unique_classes, class_counts = np.unique(train_age_labels, return_counts=True)
    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=unique_classes, 
        y=train_age_labels
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    print(f"Class distribution: {dict(zip(unique_classes, class_counts))}")
    print(f"Class weights: {class_weights}")
    
    # Create data loaders for linear probe
    train_dataset = TensorDataset(
        torch.tensor(train_embeddings, dtype=torch.float32),
        torch.tensor(train_age_labels, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(val_embeddings, dtype=torch.float32),
        torch.tensor(val_age_labels, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(test_embeddings, dtype=torch.float32),
        torch.tensor(test_age_labels, dtype=torch.long)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=probe_config['linear_probe']['batch_size'], 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=probe_config['linear_probe']['batch_size'], 
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=probe_config['linear_probe']['batch_size'], 
        shuffle=False
    )
    
    # Initialize linear probe model
    linear_model = LinearProbeModel(
        input_dim=1024,
        num_classes=probe_config['linear_probe']['num_classes'],
        learning_rate=probe_config['linear_probe']['learning_rate']
    )
    
    # Setup TensorBoard logger
    logger = TensorBoardLogger(
        save_dir=probe_config['log_dir'],
        name="linear_probe_age"
    )
    
    # Setup checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        dirpath=probe_config['checkpoints_dir'],
        filename="best_linear_probe-{epoch:02d}-{val_loss:.2f}",
        verbose=True
    )
    
    # Setup trainer
    trainer = Trainer(
        max_epochs=probe_config['linear_probe']['max_epochs'],
        callbacks=[checkpoint_callback],
        devices=1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        log_every_n_steps=10,
        logger=logger
    )
    
    # Train the linear probe
    print("Training linear probe...")
    trainer.fit(linear_model, train_loader, val_loader)
    
    # Load best model and test
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    best_model = LinearProbeModel.load_from_checkpoint(checkpoint_callback.best_model_path)
    
    print("Evaluating on test set...")
    results = trainer.test(best_model, test_loader)
    
    # Get the AUC from the model's stored value
    test_auc = best_model.test_auc
    
    # Save final results
    results_file = os.path.join(probe_config['output_dir'], probe_config['results_file'])
    with open(results_file, 'w') as f:
        f.write("Linear Probe Age Group Analysis Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Model: {probe_config['checkpoint_path']}\n")
        f.write(f"Test Macro AUC: {test_auc:.4f}\n")
        f.write(f"Class distribution: {dict(zip(unique_classes, class_counts))}\n")
        f.write(f"Class weights: {class_weights}\n")
        f.write(f"\nPer-class AUC scores:\n")
        f.write(f"Class 0 (0-36): {best_model.auc_per_class[0]:.4f}\n")
        f.write(f"Class 1 (36-50): {best_model.auc_per_class[1]:.4f}\n")
        f.write(f"Class 2 (50-65): {best_model.auc_per_class[2]:.4f}\n")
        f.write(f"Class 3 (65+): {best_model.auc_per_class[3]:.4f}\n")
    
    print(f"Results saved to: {results_file}")
    print("Linear probe analysis completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Linear Probe for Age Group Analysis')
    parser.add_argument('--config', default='linear_probe_config.yaml', 
                       help='Path to linear probe configuration file')
    args = parser.parse_args()
    
    main(args)
