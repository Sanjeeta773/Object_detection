import torch
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torchvision import transforms
from model import get_model
from args import get_args

def evaluate_on_new_images():
    args = get_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    

    #  Find images (Catch-all for any extension)
    new_image_paths = glob.glob(os.path.join(args.new_images_dir, '*'))
    

    #  load images
    valid_images = [f for f in new_image_paths if f.lower().endswith(('.png', '.jpg', '.jpeg'))]


    # Load model
    model = get_model(num_classes=args.num_classes)
    if not os.path.exists(args.model_path):
        print(f"CRITICAL ERROR: Model file {args.model_path} not found!")
        return
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    

    transform = transforms.Compose([transforms.ToTensor()])

    for img_path in valid_images:
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)

        boxes = outputs[0]['boxes'].cpu().numpy()
        scores = outputs[0]['scores'].cpu().numpy()

        

        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(img)

        for i, box in enumerate(boxes):
            if scores[i] > 0.6: 
                xmin, ymin, xmax, ymax = box
                rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                         linewidth=3, edgecolor='red', facecolor='none')
                ax.add_patch(rect)

                ax.text(xmin, ymin-10, f"Handle: {scores[i]:.2f}", 
                color='white', bbox=dict(facecolor='red', alpha=0.5))

        plt.title(f"Detection Results: {os.path.basename(img_path)}")
        plt.axis('off')
        plt.show() 
        

if __name__ == "__main__":
    evaluate_on_new_images()