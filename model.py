import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision import models
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from tqdm import tqdm
import timm
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import math


def make_identity_linear(in_features):
    """Create an identity linear layer for head removal."""
    layer = nn.Linear(in_features, in_features)
    nn.init.eye_(layer.weight)
    if layer.bias is not None:
        layer.bias.data.zero_()
    return layer


def freeze_model(model):
    """Freeze all parameters of a model."""
    for param in model.parameters():
        param.requires_grad = False


class BaseModel(pl.LightningModule):
    """Base class with common age group encoding functionality."""
    
    def __init__(self):
        super().__init__()
        # Initialize age group encoder
        self.age_group_encoder = None
        self.age_group_classes = None

    def fit_age_group_encoder(self, dataloader):
        """Fit the age group encoder using the training data."""
        all_age_groups = []
        for batch in dataloader:
            age_groups = batch['age_group']
            all_age_groups.extend(age_groups)
        
        # Create and fit label encoder
        self.age_group_encoder = LabelEncoder()
        self.age_group_encoder.fit(all_age_groups)
        self.age_group_classes = self.age_group_encoder.classes_
        print(f"Age group encoder fitted with classes: {self.age_group_classes}")

    def encode_age_groups(self, age_groups):
        """Convert age groups to one-hot encoded tensors."""
        if self.age_group_encoder is None:
            # If encoder not fitted, create a simple mapping
            unique_groups = sorted(list(set(age_groups)))
            self.age_group_encoder = LabelEncoder()
            self.age_group_encoder.fit(unique_groups)
            self.age_group_classes = self.age_group_encoder.classes_
            print(f"Age group encoder created with classes: {self.age_group_classes}")
        
        # Use pre-computed one-hot encodings for better performance
        if not hasattr(self, '_age_group_one_hot_cache'):
            self._age_group_one_hot_cache = {}
        
        # Create cache key from age groups
        cache_key = tuple(age_groups)
        if cache_key in self._age_group_one_hot_cache:
            return self._age_group_one_hot_cache[cache_key]
        
        # Encode to integers
        encoded = self.age_group_encoder.transform(age_groups)
        # Convert to one-hot
        num_classes = len(self.age_group_classes)
        one_hot = torch.zeros(len(age_groups), num_classes)
        one_hot[torch.arange(len(age_groups)), encoded] = 1
        
        # Cache the result
        self._age_group_one_hot_cache[cache_key] = one_hot
        return one_hot

    def unpack_batch(self, batch):
        return batch['image'], batch['label'], batch['age_group'], batch['age'], batch['age_group_one_hot']


# --- add at top of model.py ---
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

def grl(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

def schedule_lambda(global_step, total_steps, lambda_max=0.5):
    """
    Ganin-style schedule for adversarial lambda.
    Gradually increases from ~0 at step 0 to lambda_max at the end.
    More robust implementation with bounds.
    """
    p = max(0.0, min(1.0, global_step / max(1, total_steps)))
    return lambda_max * (2.0 / (1.0 + math.exp(-10 * p)) - 1.0)

class AgeAdversary(nn.Module):
    """
    Separate age adversary with configurable capacity and dropout.
    """
    def __init__(self, input_dim, num_age_groups, hidden_dim=256, dropout=0.2, mode="categorical"):
        super().__init__()
        self.mode = mode
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_age_groups if mode == "categorical" else (num_age_groups - 1))
        )
    
    def forward(self, features):
        return self.mlp(features)

def ordinal_loss(logits, age_targets):
    """
    Ordinal loss for age groups using cumulative logits.
    """
    B, K1 = logits.shape
    K = K1 + 1
    # Build cumulative binary targets from integer bins
    y = F.one_hot(age_targets.clamp(0, K-1), num_classes=K)[:, 1:].float()
    return F.binary_cross_entropy_with_logits(logits, y, reduction='mean')

# --------------------------------

class DenseNetAgeAdv(BaseModel):
    """
    DenseNet backbone with:
      - multilabel disease head (14 logits)
      - age-group adversary on GAP features via GRL
      - improved architecture with separate AgeAdversary
    """
    def __init__(self, num_classes=14, num_age_groups=4, adv_age_lambda=0.5, 
                 use_scheduled_lambda=True, age_mode="categorical", age_head_hidden=256, 
                 age_head_dropout=0.2, debias_enable=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_age_groups = num_age_groups
        self.adv_age_lambda = adv_age_lambda
        self.use_scheduled_lambda = use_scheduled_lambda
        self.age_mode = age_mode
        self.debias_enable = debias_enable
        
        # Initialize logging attributes
        self.current_lambda = 0.0
        self.last_age_logits = None
        self.last_age_targets = None

        # Backbone
        backbone = models.densenet121(pretrained=True)
        # Keep feature extractor (up to norm+relu+avgpool) and drop classifier
        self.features = backbone.features  # conv blocks
        self.norm = nn.BatchNorm2d(1024)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        feat_dim = 1024
        # Main multilabel head (14 diseases)
        self.cls_head = nn.Linear(feat_dim, self.num_classes)
        # Age-group adversary (separate class with configurable capacity)
        self.age_adversary = AgeAdversary(
            input_dim=feat_dim,
            num_age_groups=self.num_age_groups,
            hidden_dim=age_head_hidden,
            dropout=age_head_dropout,
            mode=age_mode
        )

        self.bce_logits = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss()

    def extract_gap_feat(self, x):
        x = self.features(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.avgpool(x)           # B x 1024 x 1 x 1
        x = torch.flatten(x, 1)       # B x 1024
        return x

    def forward(self, x):
        # For eval/test convenience: return only disease logits
        feat = self.extract_gap_feat(x)
        logits = self.cls_head(feat)
        return logits

    def remove_head(self):
        # When extracting embeddings, expose GAP features as the output
        feat_dim = 1024
        # tiny identity linear to stay compatible with existing code paths
        device = next(self.parameters()).device  # Get the device of the model
        self.cls_head = make_identity_linear(feat_dim).to(device)

    def process_batch(self, batch, current_step=None, total_steps=None):
        img, lab, age_group, age, age_onehot = self.unpack_batch(batch)  # provided by BaseModel
        device = img.device

        # Calculate current lambda value
        if not self.debias_enable:
            current_lambda = 0.0
        elif self.use_scheduled_lambda and current_step is not None and total_steps is not None:
            current_lambda = schedule_lambda(current_step, total_steps, self.adv_age_lambda)
        else:
            current_lambda = self.adv_age_lambda

        # disease head
        feat = self.extract_gap_feat(img)
        logits_y = self.cls_head(feat)
        loss_cls = self.bce_logits(logits_y, lab)

        # age adversary: GRL on the same GAP feat
        # convert provided one-hot to indices (dataset already aligns one-hot to label order)
        age_targets = torch.argmax(age_onehot.to(device), dim=1)
        
        # Use separate age adversary with GRL
        # Always run through GRL - when λ=0, it passes zero gradient to features
        # but age head can still learn and warm up properly
        h_adv = grl(feat, current_lambda)
        age_logits = self.age_adversary(h_adv)
        
        # Choose loss based on age mode
        if self.age_mode == "categorical":
            loss_age = self.ce(age_logits, age_targets)
        else:  # ordinal
            loss_age = ordinal_loss(age_logits, age_targets)

        # CORRECT loss calculation: main loss + λ * age loss
        # GRL already flips gradients for features, so we add age loss
        # This makes: feature extractor tries to maximize age loss (via GRL)
        # while age adversary tries to minimize age loss
        total = loss_cls + current_lambda * loss_age
        return total, loss_cls.detach(), loss_age.detach(), current_lambda

    # Lightning hooks
    def training_step(self, batch, batch_idx):
        # Calculate current step and total steps for lambda scheduling
        current_step = self.global_step
        total_steps = self.trainer.estimated_stepping_batches
        
        total, loss_cls, loss_age, current_lambda = self.process_batch(batch, current_step, total_steps)
        
        # Store current lambda for logging callbacks
        self.current_lambda = current_lambda
        
        # Store age logits and targets for accuracy calculation
        img, lab, age_group, age, age_onehot = self.unpack_batch(batch)
        device = img.device
        feat = self.extract_gap_feat(img)
        h_adv = grl(feat, current_lambda)
        age_logits = self.age_adversary(h_adv)
        age_targets = torch.argmax(age_onehot.to(device), dim=1)
        
        self.last_age_logits = age_logits.detach()
        self.last_age_targets = age_targets.detach()
        
        # Log metrics
        self.log('train_total', total, prog_bar=True)
        self.log('train_cls', loss_cls)
        self.log('train_age_adv', loss_age)
        self.log('current_lambda', current_lambda, prog_bar=True)
        
        if batch_idx == 0:
            grid = torchvision.utils.make_grid(batch['image'][0:4, ...], nrow=2, normalize=True)
            self.logger.experiment.add_image('training_images', grid, self.current_epoch)
        
        # Return dict for enhanced logging
        return {
            'loss': total,  # Required by PyTorch Lightning
            'train_total': total,
            'train_loss': loss_cls,
            'train_age_adv': loss_age,
            'current_lambda': current_lambda
        }

    def validation_step(self, batch, batch_idx):
        total, loss_cls, loss_age, current_lambda = self.process_batch(batch)
        self.log('val_loss', total, prog_bar=True)
        self.log('val_cls', loss_cls)
        self.log('val_age_adv', loss_age)
        self.log('val_lambda', current_lambda)

    def test_step(self, batch, batch_idx):
        total, loss_cls, loss_age, current_lambda = self.process_batch(batch)
        self.log('test_loss', total)
        self.log('test_cls', loss_cls)
        self.log('test_age_adv', loss_age)

    def configure_optimizers(self):
        return torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-3)



class ResNet(BaseModel):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.model = models.resnet34(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, self.num_classes)

    def remove_head(self):
        # Use identity nn.Linear to avoid linter/type errors
        num_features = self.model.fc.in_features
        self.model.fc = make_identity_linear(num_features)

    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        params_to_update = []
        for param in self.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
        optimizer = torch.optim.Adam(params_to_update, lr=0.001)
        return optimizer

    def process_batch(self, batch):
        img, lab, age_group, age, age_group_one_hot = self.unpack_batch(batch)
        
        # Use pre-computed one-hot encoding (already on correct device)
        age_group_one_hot = age_group_one_hot.to(img.device)
        
        out = self.forward(img)
        prob = torch.sigmoid(out)
        loss = F.binary_cross_entropy(prob, lab)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('train_loss', loss)
        # Log images to TensorBoard
        if batch_idx == 0:  # Log only first batch of each epoch
            grid = torchvision.utils.make_grid(batch['image'][0:4, ...], nrow=2, normalize=True)
            self.logger.experiment.add_image('training_images', grid, self.current_epoch)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('test_loss', loss)


class VisionTransformer(BaseModel):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.model = timm.create_model('vit_base_patch16_224', pretrained=False)
        self.model.head = nn.Linear(self.model.head.in_features, self.num_classes)

    def remove_head(self):
        num_features = self.model.head.in_features
        self.model.head = nn.Identity()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        params_to_update = []
        for param in self.parameters():
            if param.requires_grad:
                params_to_update.append(param)
        optimizer = torch.optim.Adam(params_to_update, lr=0.001)
        return optimizer

    def process_batch(self, batch):
        img, lab, age_group, age, age_group_one_hot = self.unpack_batch(batch)
        
        # Use pre-computed one-hot encoding (already on correct device)
        age_group_one_hot = age_group_one_hot.to(img.device)
        
        out = self.forward(img)
        prob = torch.sigmoid(out)
        loss = F.binary_cross_entropy(prob, lab)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('train_loss', loss)
        # Log images to TensorBoard
        if batch_idx == 0:  # Log only first batch of each epoch
            grid = torchvision.utils.make_grid(batch['image'][0:4, ...], nrow=2, normalize=True)
            self.logger.experiment.add_image('training_images', grid, self.current_epoch)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('test_loss', loss)


class DenseNet(BaseModel):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.model = models.densenet121(pretrained=True)
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, self.num_classes)

    def remove_head(self):
        # Use identity nn.Linear to avoid linter/type errors
        num_features = self.model.classifier.in_features
        # Get the device of the current classifier layer
        device = self.model.classifier.weight.device
        self.model.classifier = make_identity_linear(num_features).to(device)

    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        params_to_update = []
        for param in self.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
        optimizer = torch.optim.Adam(params_to_update, lr=0.001)
        return optimizer

    def process_batch(self, batch):
        img, lab, age_group, age, age_group_one_hot = self.unpack_batch(batch)
        
        # Use pre-computed one-hot encoding (already on correct device)
        age_group_one_hot = age_group_one_hot.to(img.device)
        
        out = self.forward(img)
        prob = torch.sigmoid(out)
        loss = F.binary_cross_entropy(prob, lab)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('train_loss', loss)
        # Log images to TensorBoard
        if batch_idx == 0:  # Log only first batch of each epoch
            grid = torchvision.utils.make_grid(batch['image'][0:4, ...], nrow=2, normalize=True)
            self.logger.experiment.add_image('training_images', grid, self.current_epoch)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('test_loss', loss)


def test(model, data_loader, device, num_classes):
    """Evaluate model on test data and return predictions, targets, logits, and paths."""
    model.eval()
    logits = []
    preds = []
    targets = []
    paths = []

    with torch.no_grad():
        for index, batch in enumerate(tqdm(data_loader, desc='Test-loop')):
            img = batch['image'].to(device)
            lab = batch['label'].to(device)
            path = batch['image_path']
            # age_group, age, age_group_one_hot are also available but not used in test
            out = model(img)
            pred = torch.sigmoid(out)
            logits.append(out)
            preds.append(pred)
            targets.append(lab)
            paths.extend(path)

        logits = torch.cat(logits, dim=0)
        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)

        counts = []
        for i in range(0, num_classes):
            t = targets[:, i] == 1
            c = torch.sum(t)
            counts.append(c)
        print(counts)

    return preds.cpu().numpy(), targets.cpu().numpy(), logits.cpu().numpy(), paths


def embeddings(model, data_loader, device):
    """Extract embeddings from model."""
    model.eval()

    embeds = []
    targets = []
    paths = []

    with torch.no_grad():
        for index, batch in enumerate(tqdm(data_loader, desc='Test-loop')):
            img = batch['image'].to(device)
            lab = batch['label'].to(device)
            path = batch['image_path']
            # age_group, age, age_group_one_hot are also available but not used in embeddings
            emb = model(img)
            embeds.append(emb.cpu())  # Move embeddings to CPU
            targets.append(lab.cpu())  # Move targets to CPU
            paths.extend(path)

    embeds = torch.cat(embeds, dim=0)
    targets = torch.cat(targets, dim=0)

    return embeds.numpy(), targets.numpy(), paths


def get_latest_checkpoint(checkpoint_dir):
    """Get the most recent checkpoint file from the directory."""
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_files = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.ckpt'):
            checkpoint_files.append(os.path.join(checkpoint_dir, file))
    
    if not checkpoint_files:
        return None
    
    # Sort by modification time to get the most recent
    checkpoint_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return checkpoint_files[0]


def cleanup_old_checkpoints(checkpoint_dir, keep_last_n=5):
    """Clean up old checkpoint files, keeping only the most recent N."""
    if not os.path.exists(checkpoint_dir):
        return
    
    checkpoint_files = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.ckpt'):
            checkpoint_files.append(os.path.join(checkpoint_dir, file))
    
    if len(checkpoint_files) <= keep_last_n:
        return
    
    # Sort by modification time
    checkpoint_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Remove old checkpoints
    for old_checkpoint in checkpoint_files[keep_last_n:]:
        try:
            os.remove(old_checkpoint)
            print(f"Removed old checkpoint: {old_checkpoint}")
        except Exception as e:
            print(f"Failed to remove checkpoint {old_checkpoint}: {e}")


def calculate_aucs(preds, targets, num_classes):
    """Calculate AUC scores for each class and macro AUC."""
    aucs = []
    for i in range(num_classes):
        try:
            auc = roc_auc_score(targets[:, i], preds[:, i])
        except ValueError:
            auc = float('nan')
        aucs.append(auc)
        print(f"AUC Class {i}: {auc:.4f}")
    
    macro_auc = np.nanmean(aucs)
    print(f"Macro AUC: {macro_auc:.4f}")
    
    return aucs, macro_auc


# Model registry for easy model selection
MODEL_REGISTRY = {
    'DenseNet': DenseNet,
    'ResNet': ResNet,
    'VisionTransformer': VisionTransformer,
    'DenseNetAgeAdv': DenseNetAgeAdv,   # <-- new
}
