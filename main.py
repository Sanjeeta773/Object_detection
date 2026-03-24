import pandas as pd 
import os
import torch
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
    
    # Training-Validation Cycle
    train_model(model, train_loader, val_loader, device, args)

if __name__ == "__main__":
    main()