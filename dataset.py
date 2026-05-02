import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class RxHandBDDataset(Dataset):
    def __init__(self, csv_path, image_dir):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = os.path.join(self.image_dir, row["Images"])
        image = Image.open(image_path).convert("RGB")

        text = row["Text"]

        return image, text