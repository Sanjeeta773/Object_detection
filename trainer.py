import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils import show_batch

def validate_model(model, val_loader, device):
    model.train() # Required for Faster R-CNN loss
    val_loss_sum = 0.0
    val_count = 0
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            loss = sum(l for l in loss_dict.values())
            
            val_loss_sum += loss.item() * len(images)
            val_count += len(images)
            
    return val_loss_sum / val_count if val_count > 0 else 0


def save_learning_curve(train_history, val_history): #function for plotting learning curve
        epochs = range(1, len(train_history) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_history, label='Training Loss', marker='o', color='blue')
        plt.plot(epochs, val_history, label='Validation Loss', marker='s', color='orange')
    
        plt.title('Learning Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.xticks(epochs)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
    
        plt.savefig('learning_curve_automated.png')
        plt.show()
        print("Graph saved automatically as learning_curve_automated.png")

def train_model(model, train_loader, val_loader, device, args):
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)
    best_val_loss = float('inf')
    
    train_history =[]
    val_history = []


    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            
            # Visualize the first batch of the first epoch
            if epoch == 0 and batch_idx == 0:
                print("Visualizing the first training batch...")
                show_batch(images, targets)

            #  Move data to device
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # OPTIMIZATION STEP
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            loss = sum(l for l in loss_dict.values())
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * len(images)

            print(f"Epoch {epoch+1} | Batch {batch_idx + 1}/{len(train_loader)} | Loss: {loss.item():.4f}")
            

        train_epoch_loss = running_loss / len(train_loader.dataset)
        val_loss = validate_model(model, val_loader, device)

        train_history.append(train_epoch_loss)
        val_history.append(val_loss)
      
        print(f"--- Epoch {epoch + 1} Result: Train Loss: {train_epoch_loss:.4f} | Val Loss: {val_loss:.4f} ---")
            
    
        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
            print(" New Best Model Saved!")

    save_learning_curve(train_history, val_history)        

    return{"train_loss": train_history, "val_loss": val_history}