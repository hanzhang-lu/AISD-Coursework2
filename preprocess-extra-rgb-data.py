import numpy as np
import os
from PIL import Image
import cv2

#
# Pre-Process Extra Dataset (Landcover.ai)
#

base_dir = "/root/autodl-tmp/attention-mechanism-unet/extra_dataset"

# Input directory - preprocessed 512x512 tiles from split.py
output_dir = os.path.join(base_dir, "output")

# Split files
train_split_file = os.path.join(base_dir, "train.txt")
val_split_file = os.path.join(base_dir, "val.txt")
test_split_file = os.path.join(base_dir, "test.txt")

def load_split_data(split_file):
    """Load tile names from split file."""
    with open(split_file, 'r') as f:
        # Each line is in format: BASENAME_N (e.g., M-33-20-D-c-4-2_0)
        # These correspond to files in output dir: BASENAME_N.jpg and BASENAME_N_m.png
        lines = [line.strip() for line in f.readlines() if line.strip()]
    return lines

def load_image_mask_pairs(tile_names):
    """Load preprocessed 512x512 image and mask tiles."""
    images = []
    masks = []
    image_names = []
    mask_names = []
    
    for tile_name in tile_names:
        # Files are already 512x512 from split.py
        img_path = os.path.join(output_dir, f"{tile_name}.jpg")
        mask_path = os.path.join(output_dir, f"{tile_name}_m.png")
        
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            print(f"[WARNING] Missing tile pair: {img_path} or {mask_path}")
            continue
            
        try:
            # Load using cv2 (same as split.py used for saving)
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path)
            
            # Convert BGR to RGB for images
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            img = Image.fromarray(img)
            mask = Image.fromarray(mask)
            
            images.append(img)
            masks.append(mask)
            image_names.append(f"{tile_name}.jpg")
            mask_names.append(f"{tile_name}_m.png")
            
        except Exception as e:
            print(f"[ERROR] Error loading {img_path} or {mask_path}: {str(e)}")
    
    return images, masks, image_names, mask_names

# -------------------------
# 1. Load training data
# -------------------------
print("[INFO] Loading training data...")
train_tile_names = load_split_data(train_split_file)
training_images, training_masks, training_image_names, training_mask_names = load_image_mask_pairs(train_tile_names)
print(f"[INFO] Loaded {len(training_images)} training pairs.")


# -------------------------
# 2. Load validation data
# -------------------------
print("[INFO] Loading validation data...")
val_tile_names = load_split_data(val_split_file)
validation_images, validation_masks, validation_image_names, validation_mask_names = load_image_mask_pairs(val_tile_names)
print(f"[INFO] Loaded {len(validation_images)} validation pairs.")


# -------------------------
# 3. Load test data
# -------------------------
print("[INFO] Loading test data...")
test_tile_names = load_split_data(test_split_file)
test_images, test_masks, test_image_names, test_mask_names = load_image_mask_pairs(test_tile_names)
print(f"[INFO] Loaded {len(test_images)} test pairs.")


def preprocess_images(images, target_size=(512, 512), is_mask=False):
    """Preprocess images: normalize and convert to numpy arrays (already 512x512)."""
    processed = []
    for img in images:
        # Convert to RGB if not already
        if img.mode != 'RGB' and not is_mask:
            img = img.convert('RGB')
        
        # Images are already 512x512 from split.py, no resize needed
        # Verify size just in case
        if img.size != target_size:
            print(f"[WARNING] Image size {img.size} != expected {target_size}, resizing...")
            img = img.resize(target_size, Image.NEAREST if is_mask else Image.BICUBIC)
        
        # Convert to numpy array
        arr = np.array(img)
        
        # Normalize images to [0, 1]
        if not is_mask:
            arr = arr.astype("float32") / 255.0
        else:
            # For masks, ensure they're single-channel
            if len(arr.shape) > 2:
                arr = arr[:, :, 0]  # Take first channel if multi-channel
        
        # Add batch dimension
        if not is_mask:
            arr = arr.reshape(1, *target_size, 3)
        else:
            arr = arr.reshape(1, *target_size, 1)
        
        processed.append(arr)
    
    return processed

# -------------------------
# 4. Preprocess data
# -------------------------
print("[INFO] Preprocessing training data...")
training_images = preprocess_images(training_images, is_mask=False)
training_masks = preprocess_images(training_masks, is_mask=True)

print("[INFO] Preprocessing validation data...")
validation_images = preprocess_images(validation_images, is_mask=False)
validation_masks = preprocess_images(validation_masks, is_mask=True)

print("[INFO] Preprocessing test data...")
test_images = preprocess_images(test_images, is_mask=False)
test_masks = preprocess_images(test_masks, is_mask=True)


# -------------------------
# 5. Save preprocessed data as .npy files
# -------------------------
out_root = "extra-dataset-processed-5classes"

train_img_out_dir = os.path.join(out_root, "training", "images")
train_mask_out_dir = os.path.join(out_root, "training", "masks")
val_img_out_dir = os.path.join(out_root, "validation", "images")
val_mask_out_dir = os.path.join(out_root, "validation", "masks")
test_img_out_dir = os.path.join(out_root, "test", "images")
test_mask_out_dir = os.path.join(out_root, "test", "masks")

# Create output directories
os.makedirs(train_img_out_dir, exist_ok=True)
os.makedirs(train_mask_out_dir, exist_ok=True)
os.makedirs(val_img_out_dir, exist_ok=True)
os.makedirs(val_mask_out_dir, exist_ok=True)
os.makedirs(test_img_out_dir, exist_ok=True)
os.makedirs(test_mask_out_dir, exist_ok=True)

def save_data(images, masks, img_names, mask_names, img_dir, mask_dir):
    """Save images and masks to the specified directories."""
    for img_arr, mask_arr, img_name, mask_name in zip(images, masks, img_names, mask_names):
        # Save image
        img_path = os.path.join(img_dir, os.path.splitext(img_name)[0] + ".npy")
        np.save(img_path, img_arr)
        
        # Save mask
        mask_path = os.path.join(mask_dir, os.path.splitext(mask_name)[0] + ".npy")
        np.save(mask_path, mask_arr)

# Save training data
print("[INFO] Saving training data...")
save_data(training_images, training_masks, training_image_names, training_mask_names, 
          train_img_out_dir, train_mask_out_dir)

# Save validation data
print("[INFO] Saving validation data...")
save_data(validation_images, validation_masks, validation_image_names, validation_mask_names,
          val_img_out_dir, val_mask_out_dir)

# Save test data
print("[INFO] Saving test data...")
save_data(test_images, test_masks, test_image_names, test_mask_names,
          test_img_out_dir, test_mask_out_dir)

print(f"[INFO] Preprocessing finished. Saved npy files under '{out_root}/'.")
print("\nDataset statistics:")
print(f"- Training images: {len(training_images)}")
print(f"- Validation images: {len(validation_images)}")
print(f"- Test images: {len(test_images)}")
