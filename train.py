import os
import torch
import torch.nn as nn
import pandas as pd
import pytorch_lightning as pl
from datetime import datetime
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from argparse import ArgumentParser
import yaml
import csv
import time
import math
from collections import defaultdict, deque

# Import from our modular files
from dataset import CheXpertDataModule
from model import (
    MODEL_REGISTRY, test, embeddings, get_latest_checkpoint, 
    cleanup_old_checkpoints, calculate_aucs, make_identity_linear
)

# =============================================================================
# COMPREHENSIVE LOGGING SYSTEM FOR ADVERSARIAL TRAINING
# =============================================================================

class EMA:
    """Exponential Moving Average for smoothing metrics"""
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.state = {}
    
    def update(self, key, value):
        if value is None: return None
        v = float(value)
        if key not in self.state: 
            self.state[key] = v
        else: 
            self.state[key] = self.alpha * v + (1 - self.alpha) * self.state[key]
        return self.state[key]
    
    def get(self, key, default=None): 
        return self.state.get(key, default)

def _ensure_dir(path):
    if path and not os.path.exists(path): 
        os.makedirs(path, exist_ok=True)

def adversary_accuracy_from_logits(logits, targets, mode="categorical"):
    """
    Compute adversary accuracy from logits
    logits: torch.Tensor (B,C) for categorical or (B,K-1) for ordinal (cumulative)
    targets: torch.LongTensor age bin indices [0..C-1] or [0..K-1]
    Returns a float accuracy (categorical) or negative MAE proxy (ordinal).
    """
    import torch.nn.functional as F
    if mode == "categorical":
        preds = logits.argmax(dim=1)
        return (preds == targets).float().mean().item()
    else:
        # For ordinal, convert cumulative logits to bin by counting thresholds passed
        probs_gt = torch.sigmoid(logits)  # P(y > k)
        # Predict bin as number of thresholds where prob>0.5
        pred_bins = (probs_gt > 0.5).sum(dim=1)
        mae = (pred_bins.to(torch.float32) - targets.to(torch.float32)).abs().mean().item()
        # Return "accuracy-like" score: higher is better => 1 - normalized MAE
        K_minus_1 = logits.shape[1]
        return max(0.0, 1.0 - mae / max(1, K_minus_1))

class AdvLogger:
    """
    Comprehensive logging system for adversarial training
    Tracks and logs:
      - main_loss, age_loss, total_loss
      - lambda (GRL strength)
      - adv_acc (age head accuracy or MAE depending on mode)
      - main_auc_macro, main_auc_micro
      - per_label_auc (list)
      - probe_age_auc (frozen-probe leakage)
      - (optional) ece_macro
    Writes to TensorBoard + CSV.
    """
    def __init__(self, log_dir="logs/adv", csv_name="train_log.csv", per_label_names=None):
        ts = time.strftime("%Y%m%d-%H%M%S")
        self.run_dir = os.path.join(log_dir, ts)
        _ensure_dir(self.run_dir)
        self.csv_path = os.path.join(self.run_dir, csv_name)
        self.tb = None  # Will be set by TensorBoardLogger
        self.ema = EMA(alpha=0.1)
        self.per_label_names = per_label_names or []
        self._init_csv()

        # rolling buffers (last N steps) for quick sanity checks
        self.buffers = defaultdict(lambda: deque(maxlen=200))

    def _init_csv(self):
        self.csv_fields = [
            "step","epoch",
            "main_loss","age_loss","total_loss",
            "lambda","adv_acc",
            "main_auc_macro","main_auc_micro","probe_age_auc","ece_macro",
        ]
        # add dynamic per-label AUC fields if provided
        for i, name in enumerate(self.per_label_names):
            self.csv_fields.append(f"auc_{i}_{name}" if name else f"auc_{i}")
        with open(self.csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.csv_fields).writeheader()

    def set_tensorboard(self, tb_logger):
        """Set TensorBoard logger for logging"""
        self.tb = tb_logger.experiment if hasattr(tb_logger, 'experiment') else tb_logger

    # --- helpers you can call per-step ---
    def log_step(self, step, epoch, main_loss=None, age_loss=None, total_loss=None,
                 lambd=None, adv_acc=None):
        # Keep short-term buffers + EMAs
        for k, v in [("main_loss", main_loss), ("age_loss", age_loss),
                     ("total_loss", total_loss), ("lambda", lambd), ("adv_acc", adv_acc)]:
            if v is not None:
                self.buffers[k].append(float(v))
                self.ema.update(k, v)
                if self.tb:
                    self.tb.add_scalar(f"train/{k}", float(v), step)

    # --- call once per epoch with evaluation metrics ---
    def log_epoch(self, step, epoch,
                  main_auc_macro=None, main_auc_micro=None,
                  per_label_auc=None, probe_age_auc=None, ece_macro=None):
        row = {
            "step": step, "epoch": epoch,
            "main_loss": self.ema.get("main_loss"),
            "age_loss": self.ema.get("age_loss"),
            "total_loss": self.ema.get("total_loss"),
            "lambda": self.ema.get("lambda"),
            "adv_acc": self.ema.get("adv_acc"),
            "main_auc_macro": self._to_float(main_auc_macro),
            "main_auc_micro": self._to_float(main_auc_micro),
            "probe_age_auc": self._to_float(probe_age_auc),
            "ece_macro": self._to_float(ece_macro),
        }

        # TensorBoard scalars
        if self.tb:
            if main_auc_macro is not None: 
                self.tb.add_scalar("eval/main_auc_macro", float(main_auc_macro), step)
            if main_auc_micro is not None: 
                self.tb.add_scalar("eval/main_auc_micro", float(main_auc_micro), step)
            if probe_age_auc is not None: 
                self.tb.add_scalar("eval/probe_age_auc", float(probe_age_auc), step)
            if ece_macro is not None: 
                self.tb.add_scalar("eval/ece_macro", float(ece_macro), step)

        # Per-label AUCs
        if per_label_auc is not None:
            for i, auc in enumerate(per_label_auc):
                key = f"auc_{i}_{self.per_label_names[i]}" if i < len(self.per_label_names) and self.per_label_names[i] else f"auc_{i}"
                row[key] = self._to_float(auc)
                if self.tb and auc is not None:
                    self.tb.add_scalar(f"eval/per_label/{key}", float(auc), step)

        # Write CSV row
        with open(self.csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.csv_fields).writerow(row)

    def close(self):
        if self.tb: 
            self.tb.close()

    @staticmethod
    def _to_float(x):
        try:
            return None if x is None else float(x)
        except Exception:
            return None

# =============================================================================
# ENHANCED CALLBACKS WITH COMPREHENSIVE LOGGING
# =============================================================================

class AdversarialLoggingCallback(Callback):
    """Callback to log adversarial training metrics during training"""
    def __init__(self, adv_logger):
        super().__init__()
        self.adv_logger = adv_logger
        self.current_epoch = 0
        self.global_step = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Log step-level metrics during training"""
        if hasattr(pl_module, 'current_lambda'):
            # Extract metrics from the model if available
            main_loss = outputs.get('train_loss', None) if isinstance(outputs, dict) else None
            age_loss = outputs.get('train_age_adv', None) if isinstance(outputs, dict) else None
            total_loss = outputs.get('train_total', None) if isinstance(outputs, dict) else None
            current_lambda = pl_module.current_lambda
            
            # Calculate adversary accuracy if age logits are available
            adv_acc = None
            if hasattr(pl_module, 'last_age_logits') and hasattr(pl_module, 'last_age_targets'):
                adv_acc = adversary_accuracy_from_logits(
                    pl_module.last_age_logits, 
                    pl_module.last_age_targets,
                    mode=getattr(pl_module, 'age_mode', 'categorical')
                )
            
            self.adv_logger.log_step(
                step=self.global_step,
                epoch=self.current_epoch,
                main_loss=main_loss,
                age_loss=age_loss,
                total_loss=total_loss,
                lambd=current_lambda,
                adv_acc=adv_acc
            )
        
        self.global_step += 1

    def on_train_epoch_end(self, trainer, pl_module):
        """Log epoch-level evaluation metrics"""
        self.current_epoch = trainer.current_epoch
        
        # Calculate evaluation metrics
        # Note: These would need to be computed from validation/test data
        # For now, we'll log what's available from the model
        
        # Get validation metrics if available
        val_loss = trainer.callback_metrics.get('val_loss', None)
        val_cls = trainer.callback_metrics.get('val_cls', None)
        val_age_adv = trainer.callback_metrics.get('val_age_adv', None)
        
        # Log epoch metrics
        self.adv_logger.log_epoch(
            step=self.global_step,
            epoch=self.current_epoch,
            main_auc_macro=None,  # Would need to compute from validation data
            main_auc_micro=None,  # Would need to compute from validation data
            per_label_auc=None,   # Would need to compute from validation data
            probe_age_auc=None,   # Would need to compute linear probe
            ece_macro=None        # Would need to compute calibration error
        )

class TrainPredictionsAndEmbeddingsCallback(Callback):
    def __init__(self, out_dir, model_class, num_classes, train_meta, target_cols, adv_logger=None):
        super().__init__()
        self.out_dir = out_dir
        self.model_class = model_class
        self.num_classes = num_classes
        self.train_meta = train_meta
        self.target_cols = target_cols
        self.original_head = None  # To store and restore the original head
        self.adv_logger = adv_logger  # Enhanced logging

    def on_train_epoch_end(self, trainer, pl_module):
        device = pl_module.device
        dataloader = trainer.datamodule.train_dataloader()
        model = pl_module.eval()

        # --- Save predictions ---
        preds, targets, logits, paths = test(model, dataloader, device, self.num_classes)
        class_cols = pd.Index([f'class_{i}' for i in range(self.num_classes)])
        logit_cols = pd.Index([f'logit_{i}' for i in range(self.num_classes)])
        target_cols = pd.Index([f'target_{i}' for i in range(self.num_classes)])
        df = pd.DataFrame(preds, columns=class_cols)
        df['img_path'] = paths
        df_logits = pd.DataFrame(logits, columns=logit_cols)
        df_targets = pd.DataFrame(targets, columns=target_cols)
        df_all = pd.concat([df, df_logits, df_targets], axis=1)
        
        # Fix path matching for merge
        # Convert dataset paths to match metadata format
        df_all['img_path'] = df_all['img_path'].apply(lambda x: x.replace('/home/4tb/mimic/files/mimic-cxr-jpg/2.1.0/files/', '/workspace/mimic/2.1.0/files/'))
        
        train_final = pd.merge(df_all, self.train_meta, on='img_path', how='inner')
        train_final.to_csv(
            os.path.join(self.out_dir, f'predictions.train.epoch_{trainer.current_epoch}.csv'),
            index=False
        )

        # --- Save embeddings ---
        # Save and remove the head - handle different model structures
        if self.original_head is None:
            if hasattr(pl_module, 'model') and hasattr(pl_module.model, 'fc'):
                # ResNet structure
                self.original_head = pl_module.model.fc
                pl_module.model.fc = make_identity_linear(pl_module.model.fc.in_features).to(device)
            elif hasattr(pl_module, 'model') and hasattr(pl_module.model, 'classifier'):
                # DenseNet structure
                self.original_head = pl_module.model.classifier
                pl_module.model.classifier = make_identity_linear(pl_module.model.classifier.in_features).to(device)
            elif hasattr(pl_module, 'model') and hasattr(pl_module.model, 'head'):
                # VisionTransformer structure
                self.original_head = pl_module.model.head
                pl_module.model.head = nn.Identity()
            elif hasattr(pl_module, 'cls_head'):
                # DenseNetAgeAdv structure
                self.original_head = pl_module.cls_head
                pl_module.cls_head = make_identity_linear(pl_module.cls_head.in_features).to(device)
        else:
            # Remove head for embeddings
            if hasattr(pl_module, 'model') and hasattr(pl_module.model, 'fc'):
                pl_module.model.fc = make_identity_linear(pl_module.model.fc.in_features).to(device)
            elif hasattr(pl_module, 'model') and hasattr(pl_module.model, 'classifier'):
                pl_module.model.classifier = make_identity_linear(pl_module.model.classifier.in_features).to(device)
            elif hasattr(pl_module, 'model') and hasattr(pl_module.model, 'head'):
                pl_module.model.head = nn.Identity()
            elif hasattr(pl_module, 'cls_head'):
                pl_module.cls_head = make_identity_linear(pl_module.cls_head.in_features).to(device)

        embeds, labels, paths = embeddings(model, dataloader, device)
        df_emb = pd.DataFrame(embeds)
        df_emb['img_path'] = paths
        df_labels = pd.DataFrame(labels, columns=self.target_cols)
        df_all = pd.concat([df_emb, df_labels], axis=1)
        
        # Fix path matching for merge
        df_all['img_path'] = df_all['img_path'].apply(lambda x: x.replace('/home/4tb/mimic/files/mimic-cxr-jpg/2.1.0/files/', '/workspace/mimic/2.1.0/files/'))
        
        df_merged = pd.merge(df_all, self.train_meta, on='img_path', how='inner')
        df_merged.to_csv(
            os.path.join(self.out_dir, f'embeddings.train.epoch_{trainer.current_epoch}.csv'),
            index=False
        )
        # Restore model head for further training
        if self.original_head is not None:
            if hasattr(pl_module, 'model') and hasattr(pl_module.model, 'fc'):
                pl_module.model.fc = self.original_head
            elif hasattr(pl_module, 'model') and hasattr(pl_module.model, 'classifier'):
                pl_module.model.classifier = self.original_head
            elif hasattr(pl_module, 'model') and hasattr(pl_module.model, 'head'):
                pl_module.model.head = self.original_head
            elif hasattr(pl_module, 'cls_head'):
                pl_module.cls_head = self.original_head
        pl_module.train()


def main(args):
    # Load configuration
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Add save_all_checkpoints flag (default True)
    save_all_checkpoints = cfg.get('save_all_checkpoints', True)

    pl.seed_everything(42 + args.model_id, workers=True)
    out_name = f"{cfg['out_name_prefix']}_model_{args.model_id}" + datetime.now().strftime("_%Y%m%d_%H%M%S")
    out_dir = os.path.join(cfg['out_base_dir'], out_name)
    os.makedirs(out_dir, exist_ok=True)

    # Setup logging and checkpointing
    log_dir = os.path.join(cfg['log_dir'], out_name)
    checkpoint_dir = os.path.join(cfg['checkpoint_dir'], out_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Enhanced checkpoint callbacks for better failure recovery
    if save_all_checkpoints:
        save_top_k = -1  # Save all checkpoints
    else:
        save_top_k = cfg.get('checkpoint_save_top_k', 3)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", 
        mode="min", 
        save_top_k=save_top_k,  # Use flag
        dirpath=checkpoint_dir, 
        filename='{epoch:02d}-{val_loss:.2f}',
        save_last=True,  # Always save the last checkpoint
        every_n_epochs=1,  # Save every epoch
        save_on_train_epoch_end=True  # Save at the end of each training epoch
    )
    
    # Additional callback for more frequent saves during training
    frequent_checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='latest-{epoch:02d}-{step:06d}',
        save_top_k=1,
        every_n_train_steps=cfg.get('checkpoint_save_every_n_steps', 100),  # Use config value
        save_on_train_epoch_end=False
    )

    # Setup TensorBoard logger
    tensorboard_logger = TensorBoardLogger(save_dir=log_dir, name=out_name)
    tensorboard_logger.log_hyperparams(cfg)
    
    # Initialize comprehensive adversarial logging system
    adv_logger = AdvLogger(
        log_dir=os.path.join(log_dir, "adv_logs"),
        csv_name="train_log.csv",
        per_label_names=[f"class_{i}" for i in range(cfg['num_classes_main'])]
    )
    adv_logger.set_tensorboard(tensorboard_logger)

    # Validate CSV files exist
    for csv_name, csv_path in [('train', cfg['train_csv']), ('val', cfg['val_csv']), ('test', cfg['test_csv'])]:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"{csv_name} CSV file not found: {csv_path}")
        print(f"{csv_name} CSV file found: {csv_path}")
    
    # Define age groups configuration from config file
    age_groups_config = cfg.get('age_groups', [
        [0, 36, "0-36"],
        [36, 50, "36-50"],
        [50, 65, "50-65"],
        [65, -1, "65+"]
    ])
    
    # Convert config format to tuple format and handle infinity
    age_groups = []
    for min_age, max_age, group_name in age_groups_config:
        if max_age == -1:
            age_groups.append((min_age, float('inf'), group_name))
        else:
            age_groups.append((min_age, max_age, group_name))
    
    # Data module
    data = CheXpertDataModule(
        csv_train_img=cfg['train_csv'],
        csv_val_img=cfg['val_csv'],
        csv_test_img=cfg['test_csv'],
        image_size=tuple(cfg['image_size']),
        img_data_dir=cfg['img_data_dir'],
        pseudo_rgb=True,
        batch_size=cfg['batch_size_main'],
        num_workers=cfg['num_workers'],
        age_groups=age_groups
    )

    # Model
    # Dynamically choose model class from config
    model_class = MODEL_REGISTRY[cfg['model_type']]
    if args.checkpoint_path:
        if cfg['model_type'] == 'DenseNetAgeAdv':
            model = model_class.load_from_checkpoint(
                args.checkpoint_path,
                num_classes=cfg['num_classes_main'],
                num_age_groups=4,
                adv_age_lambda=cfg.get('adv_age_lambda', 0.5),
                use_scheduled_lambda=cfg.get('use_scheduled_lambda', True),
                age_mode=cfg.get('age_mode', 'categorical'),
                age_head_hidden=cfg.get('age_head_hidden', 256),
                age_head_dropout=cfg.get('age_head_dropout', 0.2),
                debias_enable=cfg.get('debias_enable', True)
            )
        else:
            model = model_class.load_from_checkpoint(
                args.checkpoint_path,
                num_classes=cfg['num_classes_main']
            )
    else:
        if cfg['model_type'] == 'DenseNetAgeAdv':
            model = model_class(
                num_classes=cfg['num_classes_main'],
                num_age_groups=4,
                adv_age_lambda=cfg.get('adv_age_lambda', 0.5),
                use_scheduled_lambda=cfg.get('use_scheduled_lambda', True),
                age_mode=cfg.get('age_mode', 'categorical'),
                age_head_hidden=cfg.get('age_head_hidden', 256),
                age_head_dropout=cfg.get('age_head_dropout', 0.2),
                debias_enable=cfg.get('debias_enable', True)
            )
        else:
            model = model_class(
                num_classes=cfg['num_classes_main']
            )

    # # Load or initialize model
    # if args.checkpoint_path:
    #     model = model_class.load_from_checkpoint(args.checkpoint_path, num_classes=cfg['num_classes_main'])
    # else:
    #     model = model_class(num_classes=cfg['num_classes_main'])

    # Train
    if not args.checkpoint_path:
        # Create callbacks list
        callbacks_list = [
            checkpoint_callback, 
            frequent_checkpoint_callback,
            AdversarialLoggingCallback(adv_logger)  # Add adversarial logging
        ]
        
        # Add early stopping if enabled
        if cfg.get('enable_early_stopping', True):
            callbacks_list.append(
                EarlyStopping(
                    monitor='val_loss',
                    patience=cfg.get('early_stopping_patience', 5),
                    mode='min',
                    verbose=True
                )
            )
        
        train_pred_emb_callback = TrainPredictionsAndEmbeddingsCallback(
            out_dir=out_dir,
            model_class=model_class,
            num_classes=cfg['num_classes_main'],
            train_meta=pd.read_csv(cfg['train_details_csv']),
            target_cols=pd.Index([f'target_{i}' for i in range(cfg['num_classes_main'])]),
            adv_logger=adv_logger
        )
        callbacks_list.append(train_pred_emb_callback)

        trainer = pl.Trainer(
            callbacks=callbacks_list,
            log_every_n_steps=5,
            max_epochs=cfg['epochs_main'],
            devices=1,
            accelerator='gpu',
            logger=tensorboard_logger,
            enable_checkpointing=True,  # Explicitly enable checkpointing
            enable_progress_bar=True,
            enable_model_summary=True,
            deterministic=False,  # Set to True if you want reproducible results
            # Add recovery options
            reload_dataloaders_every_n_epochs=0,  # Don't reload dataloaders
            # Add gradient clipping to prevent training instability (from instruction_adv.py)
            gradient_clip_val=1.0
        )
        
        try:
            print(f"Starting training for {cfg['epochs_main']} epochs...")
            print(f"Early stopping patience: {cfg.get('early_stopping_patience', 5)}")
            print(f"Training dataset size: {len(data.train_set)}")
            print(f"Validation dataset size: {len(data.val_set)}")
            print(f"Test dataset size: {len(data.test_set)}")
            
            trainer.fit(model, data)
            best_ckpt = checkpoint_callback.best_model_path
            print(f"Training completed successfully. Best checkpoint: {best_ckpt}")
            
            # Clean up old checkpoints after successful training (only if not saving all)
            if not save_all_checkpoints:
                def cleanup_old_checkpoints_safe(checkpoint_dir, keep_last_n, best_ckpt, last_ckpt):
                    if not os.path.exists(checkpoint_dir):
                        return
                    checkpoint_files = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
                    # Always keep best and last
                    keep = set([best_ckpt, last_ckpt])
                    # Sort by mtime, keep the most recent N
                    checkpoint_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                    keep.update(checkpoint_files[:keep_last_n])
                    for ckpt in checkpoint_files:
                        if ckpt not in keep:
                            try:
                                os.remove(ckpt)
                                print(f"Removed old checkpoint: {ckpt}")
                            except Exception as e:
                                print(f"Failed to remove checkpoint {ckpt}: {e}")
                last_ckpt = checkpoint_callback.last_model_path
                cleanup_old_checkpoints_safe(checkpoint_dir, cfg.get('checkpoint_cleanup_keep_last', 3), best_ckpt, last_ckpt)
        except Exception as e:
            print(f"Training failed with error: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            print("Attempting to recover from latest checkpoint...")
            
            # Try to find the most recent checkpoint
            latest_ckpt = get_latest_checkpoint(checkpoint_dir)
            
            if latest_ckpt:
                print(f"Found latest checkpoint: {latest_ckpt}")
                best_ckpt = latest_ckpt
            else:
                print("No checkpoints found. Cannot recover from failure.")
                raise e
    else:
        best_ckpt = args.checkpoint_path

    # Before loading, check if best_ckpt exists; if not, fall back to last checkpoint
    if not os.path.exists(best_ckpt):
        print(f"Best checkpoint {best_ckpt} not found, trying last checkpoint...")
        last_ckpt = checkpoint_callback.last_model_path if 'checkpoint_callback' in locals() else None
        if last_ckpt and os.path.exists(last_ckpt):
            best_ckpt = last_ckpt
            print(f"Using last checkpoint: {best_ckpt}")
        else:
            raise FileNotFoundError("No valid checkpoint found to load the model.")

    # Reload best model
    if cfg['model_type'] == 'DenseNetAgeAdv':
        model = model_class.load_from_checkpoint(best_ckpt, num_classes=cfg['num_classes_main'], num_age_groups=4, adv_age_lambda=cfg.get('adv_age_lambda', 0.1))
    else:
        model = model_class.load_from_checkpoint(best_ckpt, num_classes=cfg['num_classes_main'])
    device = torch.device(f"cuda:{args.dev}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Metadata
    train_meta = pd.read_csv(cfg['train_details_csv'])
    val_meta = pd.read_csv(cfg['val_details_csv'])
    test_meta = pd.read_csv(cfg['test_details_csv'])
    target_cols = pd.Index([f'target_{i}' for i in range(cfg['num_classes_main'])])

    train_pred_emb_callback = TrainPredictionsAndEmbeddingsCallback(
        out_dir=out_dir,
        model_class=model_class,
        num_classes=cfg['num_classes_main'],
        train_meta=train_meta,
        target_cols=target_cols
    )

    # Prediction columns
    class_cols = pd.Index([f'class_{i}' for i in range(cfg['num_classes_main'])])
    logit_cols = pd.Index([f'logit_{i}' for i in range(cfg['num_classes_main'])])
    target_cols = pd.Index([f'target_{i}' for i in range(cfg['num_classes_main'])])

    # VALIDATION
    print("VALIDATION")
    preds_val, targets_val, logits_val, paths_val = test(model, data.val_dataloader(), device, cfg['num_classes_main'])
    df_val = pd.DataFrame(preds_val, columns=class_cols)
    df_val['img_path'] = paths_val
    df_val_logits = pd.DataFrame(logits_val, columns=logit_cols)
    df_val_targets = pd.DataFrame(targets_val, columns=target_cols)
    df_val_all = pd.concat([df_val, df_val_logits, df_val_targets], axis=1)
    
    # Fix path matching for merge
    df_val_all['img_path'] = df_val_all['img_path'].apply(lambda x: x.replace('/home/4tb/mimic/files/mimic-cxr-jpg/2.1.0/files/', '/workspace/mimic/2.1.0/files/'))
    
    val_final = pd.merge(df_val_all, val_meta, on='img_path', how='inner')
    val_final.to_csv(os.path.join(out_dir, f'predictions.val.model_{args.model_id}.csv'), index=False)

    # TESTING
    print("TESTING")
    preds_test, targets_test, logits_test, paths_test = test(model, data.test_dataloader(), device, cfg['num_classes_main'])
    df_test = pd.DataFrame(preds_test, columns=class_cols)
    df_test['img_path'] = paths_test
    df_test_logits = pd.DataFrame(logits_test, columns=logit_cols)
    df_test_targets = pd.DataFrame(targets_test, columns=target_cols)
    df_test_all = pd.concat([df_test, df_test_logits, df_test_targets], axis=1)
    
    # Fix path matching for merge
    df_test_all['img_path'] = df_test_all['img_path'].apply(lambda x: x.replace('/home/4tb/mimic/files/mimic-cxr-jpg/2.1.0/files/', '/workspace/mimic/2.1.0/files/'))
    
    test_final = pd.merge(df_test_all, test_meta, on='img_path', how='inner')
    test_final.to_csv(os.path.join(out_dir, f'predictions.test.model_{args.model_id}.csv'), index=False)

    # AUCs
    print("AUC METRICS")
    aucs, macro_auc = calculate_aucs(preds_test, targets_test, cfg['num_classes_main'])
    
    with open(os.path.join(out_dir, 'per_class_auc.txt'), 'w') as f:
        for i, auc in enumerate(aucs):
            f.write(f"Class {i} AUC: {auc:.4f}\n")
        f.write(f"Macro AUC: {macro_auc:.4f}\n")

    # EMBEDDINGS
    print("EXTRACTING EMBEDDINGS")
    model.remove_head()
    for split_name, dataloader, meta in [
        ("val", data.val_dataloader(), val_meta),
        ("test", data.test_dataloader(), test_meta)
    ]:
        embeds, labels, paths = embeddings(model, dataloader, device)
        df_emb = pd.DataFrame(embeds)
        df_emb['img_path'] = paths
        df_labels = pd.DataFrame(labels, columns=target_cols)
        df_all = pd.concat([df_emb, df_labels], axis=1)
        
        # Fix path matching for merge
        df_all['img_path'] = df_all['img_path'].apply(lambda x: x.replace('/home/4tb/mimic/files/mimic-cxr-jpg/2.1.0/files/', '/workspace/mimic/2.1.0/files/'))
        
        df_merged = pd.merge(df_all, meta, on='img_path', how='inner')
        df_merged.to_csv(os.path.join(out_dir, f'embeddings.{split_name}.model_{args.model_id}.csv'), index=False)
        print(f"Saved {split_name} embeddings.")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', default='config.yaml', help='Path to config YAML file')
    parser.add_argument('--checkpoint_path', default=None, help='Optional checkpoint path to resume training')
    parser.add_argument('--dev', type=int, default=0, help='GPU device index')
    args = parser.parse_args()

    args.model_id = 1  # or any fixed value you want
    main(args) 