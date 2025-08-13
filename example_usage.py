#!/usr/bin/env python3
"""
Example usage script for Medical Image Classification with Adversarial Training

This script demonstrates how to:
1. Set up the configuration
2. Run training
3. Monitor progress
4. Analyze results
"""

import os
import yaml
import subprocess
import argparse
from pathlib import Path

def create_example_config():
    """Create an example configuration file."""
    config = {
        # Output and Logging Settings
        'out_name_prefix': 'example_experiment',
        'out_base_dir': './outputs',
        'log_dir': './logs',
        'checkpoint_dir': './checkpoints',
        
        # Data Paths - UPDATE THESE WITH YOUR ACTUAL PATHS
        'train_csv': '/path/to/your/train_mimic.csv',
        'val_csv': '/path/to/your/val_mimic.csv',
        'test_csv': '/path/to/your/test_mimic.csv',
        'train_details_csv': '/path/to/your/train_final.csv',
        'val_details_csv': '/path/to/your/val_final.csv',
        'test_details_csv': '/path/to/your/test_final.csv',
        'img_data_dir': '/path/to/your/images/',
        
        # Model Settings
        'model_type': 'DenseNetAgeAdv',  # Options: 'DenseNet', 'ResNet', 'VisionTransformer', 'DenseNetAgeAdv'
        'num_classes_main': 14,
        'adv_age_lambda': 0.1,  # Adversarial training weight
        
        # Data Settings
        'image_size': [224, 224],
        'batch_size_main': 64,  # Reduced for example
        'num_workers': 4,
        
        # Training Settings
        'epochs_main': 5,  # Reduced for example
        'enable_early_stopping': True,
        'early_stopping_patience': 3,
        
        # Checkpoint Settings
        'save_all_checkpoints': False,
        'checkpoint_save_top_k': 3,
        'checkpoint_save_every_n_steps': 100,
        'checkpoint_cleanup_keep_last': 3,
        
        # Age Groups Configuration
        'age_groups': [
            [0, 36, "0-36"],
            [36, 50, "36-50"],
            [50, 65, "50-65"],
            [65, -1, "65+"]
        ]
    }
    
    with open('example_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print("✅ Created example_config.yaml")
    print("⚠️  Please update the data paths in example_config.yaml before running!")

def run_training_example():
    """Run a training example."""
    print("🚀 Starting training example...")
    
    # Check if config exists
    if not os.path.exists('example_config.yaml'):
        print("❌ example_config.yaml not found. Run create_example_config() first.")
        return
    
    # Run training
    cmd = [
        'python', 'train.py',
        '--config', 'example_config.yaml',
        '--dev', '0'
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with error: {e}")
    except FileNotFoundError:
        print("❌ train.py not found. Make sure you're in the correct directory.")

def monitor_training():
    """Start TensorBoard for monitoring."""
    print("📊 Starting TensorBoard...")
    
    cmd = ['tensorboard', '--logdir', './logs', '--port', '6006']
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ TensorBoard failed: {e}")
    except FileNotFoundError:
        print("❌ TensorBoard not found. Install with: pip install tensorboard")

def main():
    parser = argparse.ArgumentParser(description='Medical Image Classification Example')
    parser.add_argument('--action', choices=['config', 'train', 'monitor'], 
                       default='config', help='Action to perform')
    
    args = parser.parse_args()
    
    if args.action == 'config':
        create_example_config()
    elif args.action == 'train':
        run_training_example()
    elif args.action == 'monitor':
        monitor_training()

if __name__ == "__main__":
    main()
