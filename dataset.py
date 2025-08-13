import os
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torchvision.transforms as T
import pytorch_lightning as pl
from skimage.io import imread
from tqdm import tqdm
from PIL import Image
from sklearn.preprocessing import LabelEncoder


def custom_collate_fn(batch):
    """Custom collate function to handle age, age group metadata, and one-hot encodings."""
    images = torch.stack([item['image'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    image_paths = [item['image_path'] for item in batch]
    age_groups = [item['age_group'] for item in batch]
    ages = [item['age'] for item in batch]
    age_group_one_hot = torch.stack([item['age_group_one_hot'] for item in batch])
    
    return {
        'image': images,
        'label': labels,
        'image_path': image_paths,
        'age_group': age_groups,
        'age': ages,
        'age_group_one_hot': age_group_one_hot
    }


class CheXpertDataset(Dataset):
    def __init__(self, csv_file_img, image_size, img_data_dir, augmentation=False, pseudo_rgb=True, age_groups=None):
        self.data = pd.read_csv(csv_file_img)
        self.image_size = image_size
        self.do_augment = augmentation
        self.pseudo_rgb = pseudo_rgb
        self.img_data_dir = img_data_dir
        
        # Default age groups if not provided
        if age_groups is None:
            self.age_groups = [
                (0, 36, '0-36'),
                (36, 50, '36-50'), 
                (50, 65, '50-65'),
                (65, float('inf'), '65+')
            ]
        else:
            self.age_groups = age_groups
            
        # Initialize age group encoder
        self.age_group_encoder = None
        self.age_group_one_hot_cache = {}

        self.labels = [
            'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
            'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
            'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
        ]

        # Basic transforms
        self.base_transform = T.Compose([
            T.Resize(self.image_size),
            T.ToTensor()
        ])

        # Augmentations (for PIL images)
        self.augment = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.RandomAffine(degrees=15, scale=(0.9, 1.1))], p=0.5),
        ])

        # Normalization
        if pseudo_rgb:
            # For pseudo-RGB, we'll normalize after repeating channels with ImageNet values
            self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        else:
            # For true grayscale, use standard grayscale normalization
            self.normalize = T.Normalize(mean=[0.5], std=[0.5])

        # Prepare samples with age group metadata
        self.samples = []
        all_age_groups = []
        
        for idx in tqdm(range(len(self.data)), desc='Loading Data'):
            img_path = os.path.join(self.img_data_dir, self.data.loc[idx, 'path_preproc'])
            img_label = np.zeros(len(self.labels), dtype='float32')
            for i, label in enumerate(self.labels):
                img_label[i] = float(self.data.loc[idx, label.strip()] == 1)
            
            # Get age and age group metadata
            age = None
            age_group = 'Unknown'
            
            if 'age' in self.data.columns:
                age = self.data.loc[idx, 'age']
                if pd.isna(age):
                    age = None
                else:
                    age = float(age)
                    # Create age group from age
                    for min_age, max_age, group_name in self.age_groups:
                        if min_age <= age < max_age:
                            age_group = group_name
                            break
            elif 'AgeGroup' in self.data.columns:
                age_group = self.data.loc[idx, 'AgeGroup']
            
            all_age_groups.append(age_group)
            
            self.samples.append({
                'image_path': img_path, 
                'label': img_label, 
                'age_group': age_group,
                'age': age
            })
        
        # Fit age group encoder and pre-compute one-hot encodings
        self._fit_age_group_encoder(all_age_groups)

    def _fit_age_group_encoder(self, all_age_groups):
        """Fit age group encoder and pre-compute one-hot encodings."""
        # Fit label encoder
        self.age_group_encoder = LabelEncoder()
        self.age_group_encoder.fit(all_age_groups)
        self.age_group_classes = self.age_group_encoder.classes_
        print(f"Age group encoder fitted with classes: {self.age_group_classes}")
        
        # Pre-compute one-hot encodings for all unique age groups
        unique_age_groups = sorted(list(set(all_age_groups)))
        for age_group in unique_age_groups:
            encoded = self.age_group_encoder.transform([age_group])[0]
            num_classes = len(self.age_group_classes)
            one_hot = torch.zeros(1, num_classes)
            one_hot[0, encoded] = 1
            self.age_group_one_hot_cache[age_group] = one_hot.squeeze(0)
        
        print(f"Pre-computed one-hot encodings for {len(unique_age_groups)} age groups")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        sample = self.samples[item]
        
        # Add error handling for missing files
        if not os.path.exists(sample['image_path']):
            raise FileNotFoundError(f"Image file not found: {sample['image_path']}")
            
        try:
            image = imread(sample['image_path'])
        except Exception as e:
            raise RuntimeError(f"Failed to load image {sample['image_path']}: {e}")

        if image.ndim == 3:
            image = image[..., 0]  # ensure grayscale

        image = Image.fromarray(image.astype(np.uint8))  # for PIL transforms

        if self.do_augment:
            image = self.augment(image)

        image = self.base_transform(image)  # ToTensor - now it's a tensor

        # Handle channel repetition and normalization based on pseudo_rgb setting
        if self.pseudo_rgb:
            # For pseudo-RGB: repeat channels first, then normalize with ImageNet values
            if image.shape[0] == 1:
                image = image.repeat(3, 1, 1)
            image = self.normalize(image)
        else:
            # For true grayscale: normalize directly with grayscale values
            image = self.normalize(image)

        label = torch.from_numpy(sample['label']).float()
        
        # Get pre-computed one-hot encoding
        age_group_one_hot = self.age_group_one_hot_cache[sample['age_group']]
        
        # Return image, label, age, age group metadata, and one-hot encoding
        return {
            'image_path': sample['image_path'], 
            'image': image, 
            'label': label,
            'age_group': sample['age_group'],
            'age': sample['age'],
            'age_group_one_hot': age_group_one_hot
        }


class CheXpertDataModule(pl.LightningDataModule):
    def __init__(self, csv_train_img, csv_val_img, csv_test_img, image_size, img_data_dir, pseudo_rgb, batch_size, num_workers, age_groups=None):

        super().__init__()
        self.csv_train_img = csv_train_img
        self.csv_val_img = csv_val_img
        self.csv_test_img = csv_test_img
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_data_dir = img_data_dir
        self.age_groups = age_groups

        self.train_set = CheXpertDataset(self.csv_train_img, self.image_size, self.img_data_dir, augmentation=True, pseudo_rgb=pseudo_rgb, age_groups=self.age_groups)
        self.val_set = CheXpertDataset(self.csv_val_img, self.image_size, self.img_data_dir, augmentation=False, pseudo_rgb=pseudo_rgb, age_groups=self.age_groups)
        self.test_set = CheXpertDataset(self.csv_test_img, self.image_size, self.img_data_dir, augmentation=False, pseudo_rgb=pseudo_rgb, age_groups=self.age_groups)

        print('#train: ', len(self.train_set))
        print('#val:   ', len(self.val_set))
        print('#test:  ', len(self.test_set))

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=custom_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=custom_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=custom_collate_fn) 