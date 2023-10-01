import pathlib
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import os

from utils import get_dir

class CGIARDataset(Dataset):
    """Pytorch data class"""
    
    # get the csv file name from the split
    split_to_csv_filename = {
        "train": "Train",
        "test": "Test"
    }
    
    columns = ["ID", "filename", "growth_stage", "extent", "season"]
    
    def __init__(self, root_dir: pathlib.Path, split: str ='train', transform=None):
        """
        Args:
            root_dir (pathlib.Path): Root directory containing all the image files.
            split (string): Split name ('train', 'test', etc.) to determine the CSV file.
            transform (callable, optional): Optional transform to be applied to the image.
        """
        self.images_dir = get_dir(root_dir) / split
        self.transform = transform

        # Determine the CSV file path based on the split
        self.df = pd.read_csv(root_dir / f'{self.split_to_csv_filename[split]}.csv')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = Image.open(self.images_dir / self.df.iloc[idx, self.columns.index("filename")]).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        extent = -1
        if self.split == "train":
            extent = self.df.iloc[idx, self.columns.index("extent")]
        
        return self.df.iloc[idx, self.columns.index("ID")], self.image, extent