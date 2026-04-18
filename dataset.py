import torch
import pandas as pd
import os
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torch.utils.data import DataLoader

class ObjDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["images"]).convert("RGB")
        w, h = img.size
        image = to_tensor(img)

        boxes, labels = [], []
        # Opening the label file and converting YOLO format (xc, yc, bw, bh) to XYXY
        with open(row["labels"]) as f:
            for line in f:
                cls, xc, yc, bw, bh = map(float, line.split())
                x1 = (xc - bw/2) * w
                y1 = (yc - bh/2) * h
                x2 = (xc + bw/2) * w
                y2 = (yc + bh/2) * h
                boxes.append([x1, y1, x2, y2])
                labels.append(int(cls) + 1) # Adding 1 because 0 is background

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))

def get_dataloaders(args):
    # Load the single dataset file mentioned
    full_df = pd.read_csv(os.path.join(args.csv_dir, 'dataset.csv'))
    
    # Split into Train (80%) and Validation (20%) for the ML Pipeline
    train_df = full_df.sample(frac=0.8, random_state=42)
    val_df = full_df.drop(train_df.index)

    train_dataset = ObjDetectionDataset(train_df)
    val_dataset = ObjDetectionDataset(val_df)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )

    return train_loader, val_loader