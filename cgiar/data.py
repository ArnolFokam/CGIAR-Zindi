from functools import lru_cache
import pathlib
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch

from cgiar.utils import get_dir

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
        self.split = split

        # Determine the CSV file path based on the split
        self.df = pd.read_csv(root_dir / f'{self.split_to_csv_filename[split]}.csv')

    def __len__(self):
        return len(self.df)

    @lru_cache(maxsize=50000)
    def __getitem__(self, idx):
        image = Image.open(self.images_dir / self.df.iloc[idx, self.columns.index("filename")]).convert('RGB')
        
        if self.transform:
            image = self._transform_image(image)
        
        extent = -1
        if self.split == "train":
            extent = self.df.iloc[idx, self.columns.index("extent")]
        
        extent = torch.FloatTensor([extent])
        return self.df.iloc[idx, self.columns.index("ID")], image, extent
    
    def _transform_image(self, image):
        return self.transform(image)
        
        
class CGIARDataset_V2(CGIARDataset):
    def __init__(self, root_dir: pathlib.Path, split: str ='train', transform=None, num_views=1):
        super().__init__(root_dir, split, transform)
        self.num_views = num_views
        
    def _transform_image(self, image):
        return [self.transform(image) for _ in range(self.num_views)]