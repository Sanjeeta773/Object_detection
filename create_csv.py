import os
import pandas as pd

image_dir = './Data/images'

label_root = './Data/labels/train' 
output_csv = './Data/CSVs/dataset.csv'

data = []

#  Get list of all images available
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
image_map = {os.path.splitext(f)[0].strip(): f for f in image_files}

print(f"--- Scanning Dataset ---")
print(f"Found {len(image_files)} images in {image_dir}")

#  Look for labels inside Data/labels/train
if os.path.exists(label_root):
    label_files = [f for f in os.listdir(label_root) if f.endswith('.txt')]
    print(f"Found {len(label_files)} labels in {label_root}")
    
    for lbl_name in label_files:
        base_name = os.path.splitext(lbl_name)[0].strip()
        
        
        if base_name in image_map:
            data.append({
                'image_path': os.path.join(image_dir, image_map[base_name]),
                'label_path': os.path.join(label_root, lbl_name)
            })
else:
    print(f"❌ Error: Folder {label_root} not found!")


df = pd.DataFrame(data)
if not df.empty:
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ Success! Linked {len(df)} pairs into {output_csv}")
else:
    print("❌ Error: 0 matches found. Check if filenames (e.g. IMG_7141) match exactly.")