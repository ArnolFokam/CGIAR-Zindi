import pathlib
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
import torch
import torch.nn as nn
from tqdm import tqdm

from cgiar.utils import get_dir


class RandomErasing(transforms.RandomErasing):
    def __init__(self, p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0, inplace=False):
        super().__init__(p, scale, ratio, value, inplace)
        
    def __call__(self, img: Image.Image):
        img = transforms.ToTensor()(img)
        img = super().__call__(img)
        img = transforms.ToPILImage()(img)
        return img
        

augmentations = {
    "RandomHorizontalFlip": transforms.RandomHorizontalFlip(p=0.5),
    "RandomVerticalFlip": transforms.RandomVerticalFlip(p=0.5),
    "RandomRotation": transforms.RandomRotation(degrees=45),
    "ColorJitter": transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    "RandomAffine": transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    "RandomPerspective": transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
    "RandomErasing": RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    "RandomGrayscale": transforms.RandomGrayscale(p=0.2),
    "RandomAffineWithResize": transforms.RandomAffine(degrees=30, translate=(0.2, 0.2), scale=(0.8, 1.2), interpolation=Image.BILINEAR),
    "RandomPosterize": transforms.RandomPosterize(bits=4),
    "RandomSolarize": transforms.RandomSolarize(threshold=128),
    "RandomEqualize": transforms.RandomEqualize(p=0.1),
    "Identity": nn.Identity(),
}


class ClassBalancedSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, class_weights):
        self.dataset = dataset
        self.class_weights = class_weights
        self.num_samples = len(self.dataset)

    def __iter__(self):
        class_indices = {extent: [] for _, _, extent in self.dataset}

        # Collect the indices of each class
        for i in range(self.num_samples):
            _, _, extent = self.dataset[i]
            class_indices[extent].append(i)

        # Create an array of sampled indices based on class probabilities
        sampled_indices = np.concatenate([
            np.random.choice(indices, size=int(self.class_weights[extent]), replace=True) 
            for extent, indices in class_indices.items()
        ])

        return iter(sampled_indices)

    def __len__(self):
        return self.num_samples


class CGIARDataset(Dataset):
    """Pytorch data class"""
    
    # get the csv file name from the split
    split_to_csv_filename = {
        "train": "Train",
        "test": "Test"
    }
    
    columns = ["ID", "filename", "growth_stage", "extent", "season"]
    
    def __init__(self, 
                 root_dir: pathlib.Path, 
                 split: str ='train', 
                 transform=None,
                 initial_size : int =512):
        """
        Args:
            root_dir (pathlib.Path): Root directory containing all the image files.
            split (string): Split name ('train', 'test', etc.) to determine the CSV file.
            transform (callable, optional): Optional transform to be applied to the image.
        """
        self.images_dir = get_dir(root_dir) / split
        self.transform = transform
        self.split = split

        # Determine the CSV file path based on the split
        self.df = pd.read_csv(root_dir / f'{self.split_to_csv_filename[split]}.csv')
        
        self.images = {}
        
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df)):
            image_path = self.images_dir / row['filename']
            image = Image.open(image_path)
            image = self._resize(image, initial_size)
            self.images[idx] = image

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        image = self.images[idx]
        
        if self.transform:
            image = self._transform_image(image)
        
        extent = -1
        if self.split == "train":
            extent = self.df.iloc[idx, self.columns.index("extent")]
        
        extent = torch.FloatTensor([extent])
        return self.df.iloc[idx, self.columns.index("ID")], image, extent
    
    def _transform_image(self, image):
        return self.transform(image)
    
    def _resize(self, image: Image, size: int):
        # Calculate the aspect ratio of the input image
        width, height = image.size
        aspect_ratio = width / height

        # Determine the new dimensions while maintaining the aspect ratio
        if aspect_ratio > 1:
            new_width = size
            new_height = int(size / aspect_ratio)
        else:
            new_height = size
            new_width = int(size * aspect_ratio)

        # Resize the image
        resized_image = image.resize((new_width, new_height))

        return resized_image
        
        
class CGIARDataset_V2(CGIARDataset):
    def __init__(self, 
                 root_dir: pathlib.Path, 
                 split: str ='train', 
                 transform=None, 
                 initial_size: int = 512, 
                 num_views=1):
        super().__init__(root_dir, split, transform, initial_size)
        self.num_views = num_views
        
    def _transform_image(self, image):
        return [self.transform(image) for _ in range(self.num_views)]
    
class CGIARDataset_V3(CGIARDataset_V2):
    def __init__(self, 
                 root_dir: pathlib.Path, 
                 split: str = 'train', 
                 transform=None, 
                 initial_size: int = 512, 
                 num_views=1):
        super().__init__(root_dir, split, transform, initial_size, num_views)
        
        if self.split == "train":
            # unique values from the "extent" 
            # column and use them as classes
            self.classes = self.df["extent"].unique()
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            self.idx_to_class = {idx: cls for idx, cls in enumerate(self.classes)}
            
            # get classes weights
            class_counts = self.df["extent"].value_counts().sort_index()
            total_samples = len(self.df)
            self.class_weights = total_samples / (class_counts * len(class_counts))
            self.class_weights = self.class_weights.to_numpy()
            
    def __getitem__(self, idx):
        index, image, extent = super().__getitem__(idx)
        
        if extent.item() != -1:
            extent = self.class_to_idx[extent.item()]
            extent = torch.LongTensor([extent])
            
        return index, image,  extent