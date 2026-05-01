import torch
import pandas as pd
import os
from PIL import Image
from PIL import ImageOps
from torchvision.transforms.functional import to_tensor
from torch.utils.data import DataLoader

class ObjDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, df, root_dir):
        self.df = df.reset_index(drop=True)
        self.df.columns = self.df.columns.str.strip()
        self.root_dir = root_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = str(row["images"])
        label_path = str(row["labels"])

        img = Image.open(img_path).convert("RGB")
        img = ImageOps.exif_transpose(img)
        w, h = img.size
        image = to_tensor(img)

        boxes, labels = [], []
        # Opening the label file and converting YOLO format (xc, yc, bw, bh) to XYXY
        with open(label_path) as f:
            for line in f:
                cls, xc, yc, bw, bh = map(float, line.split())
                x1 = (xc - bw/2) * w
                y1 = (yc - bh/2) * h
                x2 = (xc + bw/2) * w
                y2 = (yc + bh/2) * h
                boxes.append([x1, y1, x2, y2])
                labels.append(int(cls) + 1) # Adding 1 because 0 is background

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

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
    print(full_df.columns)
    print(full_df.head())

    # Split into Train (80%) and Validation (20%) for the ML Pipeline
    train_df = full_df.sample(frac=0.8, random_state=42)
    val_df = full_df.drop(train_df.index)

    train_dataset = ObjDetectionDataset(train_df, args.csv_dir)
    val_dataset = ObjDetectionDataset(val_df, args.csv_dir)

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
