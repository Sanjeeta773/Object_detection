import pandas as pd 
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

from trainer import train_model
from model import get_model
from dataset import get_dataloaders
from args import get_args


def main():
    args = get_args() 
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    
    # Model initialization
    model = get_model(num_classes=args.num_classes)
    model.to(device)
    
    # Data loading
    train_loader, val_loader = get_dataloaders(args)

    #  VISUALIZATION WITH BOXES
    images, targets = next(iter(train_loader))
    
    # 1. Prepare the image 
    sample_img = images[0].permute(1, 2, 0).cpu().numpy()
    sample_img = np.clip(sample_img, 0, 1) # Ensure values are 0-1 for display

    # 2. Setup the plot
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(sample_img)

    # 3. Get boxes for the FIRST image in this batch
    boxes = targets[0]['boxes'].cpu().numpy()
    
    for box in boxes:
        # box is [xmin, ymin, xmax, ymax]
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin
        
        # Create a red rectangle patch
        rect = patches.Rectangle((xmin, ymin), width, height, 
                                 linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

    plt.title(f"Check: Are boxes on the Door Handles? (Batch size: {args.batch_size})")
    plt.axis('off')
    plt.show() 
    # ---------------------------
    
    # Training-Validation Cycle

    history = train_model(model, train_loader, val_loader, device, args)
    
  
    df_history = pd.DataFrame(history)
    df_history.to_csv("training_history.csv", index=False) 

    

   
if __name__ == "__main__":
    main()