import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Activation, BatchNormalization, UpSampling2D, add, multiply
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from glob import glob

# ==========================================
# 0. User Configuration
# ==========================================
# Paths to your trained model weights
DEFAULT_STD_WEIGHTS = "/root/autodl-tmp/attention-mechanism-unet/unet-attention-extra-5classes.hdf5"
DEFAULT_PLUS2_WEIGHTS = "/root/autodl-tmp/attention-mechanism-unet/unet-attention-extra-5classes-plus2.hdf5"

# Test dataset directory
TEST_IMAGES_DIR = "/root/autodl-tmp/attention-mechanism-unet/extra-dataset-processed-5classes/test/images"

# Default output directory for results
DEFAULT_OUTPUT_DIR = "/root/autodl-tmp/attention-mechanism-unet/prediction_results_batch"

# ==========================================
# 1. Define Model Components
# ==========================================

def attention_block(x, gating, inter_shape, improved=False):
    """
    Attention Gate Mechanism.
    Filters the features from the encoder (x) using the gating signal from the decoder.
    """
    shape_x = K.int_shape(x)
    
    # Theta: Process input signal
    theta_x = Conv2D(inter_shape, (1,1), padding='same', kernel_initializer='he_normal')(x)
    if improved: theta_x = BatchNormalization()(theta_x)
    theta_x = MaxPooling2D((2,2))(theta_x)
    
    # Phi: Process gating signal
    phi_g = Conv2D(inter_shape, (1,1), padding='same', kernel_initializer='he_normal')(gating)
    if improved: phi_g = BatchNormalization()(phi_g)
    
    # Combine signals
    concat_xg = add([phi_g, theta_x])
    act_xg = Activation('relu')(concat_xg)
    
    # Psi: Generate attention map
    psi = Conv2D(1, (1,1), padding='same', kernel_initializer='he_normal')(act_xg)
    if improved: psi = BatchNormalization()(psi)
    
    sigmoid_xg = Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    
    # Upsample attention map to match input dimensions
    upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]), interpolation='bilinear')(sigmoid_xg)
    upsample_psi = tf.broadcast_to(upsample_psi, shape=tf.shape(x))
    
    # Apply attention coefficients to original input
    y = multiply([upsample_psi, x])
    return y

def convBlock_standard(input, filters, kernel=3):
    """
    Standard Convolution Block.
    [NOTE]: Activation is passed as an argument to Conv2D to match the layer count 
    of the pre-trained weights (reducing layer count mismatch).
    """
    conv = Conv2D(filters, kernel, padding='same', kernel_initializer='he_normal', activation='relu')(input)
    conv = Conv2D(filters, kernel, padding='same', kernel_initializer='he_normal', activation='relu')(conv)
    return conv

def convBlock_improved(input, filters, kernel=3):
    """
    Improved Convolution Block: Conv -> BN -> ReLU.
    """
    # First Layer
    conv = Conv2D(filters, kernel, padding='same', kernel_initializer='he_normal')(input)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    # Second Layer
    conv = Conv2D(filters, kernel, padding='same', kernel_initializer='he_normal')(conv)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    return conv

def bottleneck_improved(input, filters):
    """
    Deep Bottleneck Block with 3 Convolutional Layers.
    """
    # Layer 1
    conv = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(input)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    # Layer 2
    conv = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(conv)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    # Layer 3
    conv = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(conv)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    return conv

# ==========================================
# 2. Build Model Architectures
# ==========================================

def build_standard_unet(input_size=(512,512,3), num_classes=5, filter_base=16):
    """
    Builds the Standard Attention U-Net.
    Includes the specific structural logic to match saved weights.
    """
    inputs = Input(input_size)
    
    # Encoder
    c1 = convBlock_standard(inputs, filter_base)
    p1 = MaxPooling2D((2,2))(c1)
    c2 = convBlock_standard(p1, filter_base*2)
    p2 = MaxPooling2D((2,2))(c2)
    c3 = convBlock_standard(p2, filter_base*4)
    p3 = MaxPooling2D((2,2))(c3)
    c4 = convBlock_standard(p3, filter_base*8)
    p4 = MaxPooling2D((2,2))(c4)
    
    # Bottleneck
    c5 = convBlock_standard(p4, filter_base*16)
    
    # Decoder
    # Stage 4
    u4 = Conv2DTranspose(filter_base*8, (2,2), strides=(2,2), padding='same')(c5)
    att4 = attention_block(c4, c5, filter_base*8, improved=False)
    u4 = concatenate([u4, att4])
    c6 = convBlock_standard(u4, filter_base*8)
    
    # Stage 3
    u3 = Conv2DTranspose(filter_base*4, (2,2), strides=(2,2), padding='same')(c6)
    att3 = attention_block(c3, c6, filter_base*4, improved=False)
    u3 = concatenate([u3, att3])
    c7 = convBlock_standard(u3, filter_base*4)
    
    # Stage 2
    u2 = Conv2DTranspose(filter_base*2, (2,2), strides=(2,2), padding='same')(c7)
    att2 = attention_block(c2, c7, filter_base*2, improved=False)
    u2 = concatenate([u2, att2])
    c8 = convBlock_standard(u2, filter_base*2)
    
    # Stage 1 (Top Layer)
    # [NOTE]: Replicating a structural discrepancy from the training code.
    # The 'att1' block is calculated but overwritten by a direct skip connection (c1).
    u1 = Conv2DTranspose(filter_base, (2,2), strides=(2,2), padding='same')(c8)
    u1 = concatenate([u1, c1]) # Direct concatenation, skipping attention
    c9 = convBlock_standard(u1, filter_base)
    
    outputs = Conv2D(num_classes, (1,1), activation='softmax')(c9)
    return Model(inputs, outputs, name="Standard_Att_UNet")

def build_improved_unet_plus2(input_size=(512,512,3), num_classes=5, filter_base=16):
    """
    Builds the Improved Plus2 U-Net.
    Features: Batch Normalization, Deep Supervision, Deep Bottleneck.
    """
    inputs = Input(input_size)
    
    # Encoder
    c1 = convBlock_improved(inputs, filter_base)
    p1 = MaxPooling2D((2,2))(c1)
    c2 = convBlock_improved(p1, filter_base*2)
    p2 = MaxPooling2D((2,2))(c2)
    c3 = convBlock_improved(p2, filter_base*4)
    p3 = MaxPooling2D((2,2))(c3)
    c4 = convBlock_improved(p3, filter_base*8)
    p4 = MaxPooling2D((2,2))(c4)
    
    # Deep Bottleneck
    c5 = bottleneck_improved(p4, filter_base*16)
    
    # Decoder
    # Stage 4
    u4 = Conv2DTranspose(filter_base*8, (2,2), strides=(2,2), padding='same')(c5)
    u4 = BatchNormalization()(u4)
    u4 = Activation('relu')(u4)
    att4 = attention_block(c4, c5, filter_base*8, improved=True)
    u4 = concatenate([u4, att4])
    c6 = convBlock_improved(u4, filter_base*8)
    
    # Stage 3
    u3 = Conv2DTranspose(filter_base*4, (2,2), strides=(2,2), padding='same')(c6)
    u3 = BatchNormalization()(u3)
    u3 = Activation('relu')(u3)
    att3 = attention_block(c3, c6, filter_base*4, improved=True)
    u3 = concatenate([u3, att3])
    c7 = convBlock_improved(u3, filter_base*4)
    
    # Stage 2
    u2 = Conv2DTranspose(filter_base*2, (2,2), strides=(2,2), padding='same')(c7)
    u2 = BatchNormalization()(u2)
    u2 = Activation('relu')(u2)
    att2 = attention_block(c2, c7, filter_base*2, improved=True)
    u2 = concatenate([u2, att2])
    c8 = convBlock_improved(u2, filter_base*2)
    
    # Stage 1 (Top Layer)
    # [NOTE]: Replicating the structural discrepancy (Missing Attention Block 1)
    # to ensure layer counts match the saved weights (66 layers vs 72 layers).
    u1 = Conv2DTranspose(filter_base, (2,2), strides=(2,2), padding='same')(c8)
    u1 = BatchNormalization()(u1)
    u1 = Activation('relu')(u1)
    # Direct concatenation, skipping attention to match weights
    u1 = concatenate([u1, c1]) 
    c9 = convBlock_improved(u1, filter_base)
    
    # Main Output
    main_out = Conv2D(num_classes, (1,1), padding='same')(c9)
    main_out = Activation('softmax', name='main_output')(main_out)
    
    # Deep Supervision Output (from Stage 3/c7)
    aux_out = Conv2D(num_classes, (1,1), padding='same')(c7)
    aux_out = UpSampling2D(size=(4,4), interpolation='bilinear')(aux_out)
    aux_out = Activation('softmax', name='aux_output')(aux_out)
    
    return Model(inputs, [main_out, aux_out], name="Plus2_Improved_UNet")

# ==========================================
# 3. Predictor Utility Class
# ==========================================

class LandCoverPredictor:
    def __init__(self, standard_weights_path, plus2_weights_path):
        self.num_classes = 5
        self.input_size = (512, 512, 3)
        
        print("\n" + "="*50)
        print("MODEL INITIALIZATION")
        print("="*50)
        
        # 1. Initialize Standard Model
        self.model_std = build_standard_unet(self.input_size, self.num_classes)
        if os.path.exists(standard_weights_path):
            try:
                self.model_std.load_weights(standard_weights_path)
                print("[OK] Standard weights loaded.")
            except Exception as e:
                print(f"[ERROR] Standard weights load failed: {e}")
                print("Trying load_weights(by_name=True)...")
                try:
                    self.model_std.load_weights(standard_weights_path, by_name=True)
                    print("[OK] Standard weights loaded by name.")
                except:
                    print("[FATAL] Standard model is using random weights!")

        # 2. Initialize Plus2 Model
        self.model_plus2 = build_improved_unet_plus2(self.input_size, self.num_classes)
        if os.path.exists(plus2_weights_path):
            try:
                self.model_plus2.load_weights(plus2_weights_path)
                print("[OK] Plus2 weights loaded.")
            except Exception as e:
                 print(f"[ERROR] Plus2 weights load failed: {e}")
                 print("[FATAL] Plus2 model is using random weights!")
        
        # Color Map (RGB)
        self.colors = np.array([
            [255, 255, 255], # 0: Background
            [255, 0, 0],     # 1: Building
            [0, 255, 0],     # 2: Woodland
            [0, 0, 255],     # 3: Water
            [255, 255, 0]    # 4: Road
        ])
        
    def predict_single_npy(self, npy_path, save_path=None):
        print("\n" + "="*50)
        print(f"DIAGNOSING IMAGE: {npy_path}")
        print("="*50)
        
        if not os.path.exists(npy_path):
            print(f"[ERROR] File not found: {npy_path}")
            return
            
        img_arr = np.load(npy_path)
        
        # Try to load corresponding ground truth mask
        gt_mask = None
        # Derive mask path: replace 'images' with 'masks' and add '_m' before '.npy'
        mask_path = npy_path.replace('/images/', '/masks/').replace('.npy', '_m.npy')
        if os.path.exists(mask_path):
            try:
                gt_mask = np.load(mask_path)
                print(f"[OK] Ground truth mask loaded from: {mask_path}")
                print(f"Ground truth mask shape: {gt_mask.shape}")
                
                # Handle various mask dimensions
                # Remove batch dimension if present
                while len(gt_mask.shape) > 2 and gt_mask.shape[0] == 1:
                    gt_mask = gt_mask[0]
                
                # If still has extra dimensions, squeeze them
                if len(gt_mask.shape) > 2:
                    gt_mask = np.squeeze(gt_mask)
                
                # Final check: should be (H, W)
                if len(gt_mask.shape) != 2:
                    print(f"[WARN] Unexpected mask shape after processing: {gt_mask.shape}. Skipping mask.")
                    gt_mask = None
                else:
                    print(f"Processed mask shape: {gt_mask.shape}")
                    
            except Exception as e:
                print(f"[WARN] Failed to load mask: {e}")
                gt_mask = None
        else:
            print(f"[INFO] No ground truth mask found at: {mask_path}")
        
        # --- Diagnosis Step 1: Check Dimensions ---
        print(f"Original Shape: {img_arr.shape}")
        
        # Fix PyTorch format if detected: (3, H, W) -> (H, W, 3)
        if len(img_arr.shape) == 3 and img_arr.shape[0] == 3:
            print("[WARN] Detected (3, H, W) format. Transposing to (H, W, 3)...")
            img_arr = np.transpose(img_arr, (1, 2, 0))
            
        # Remove existing Batch dimension if present
        if len(img_arr.shape) == 4:
            img_arr = img_arr[0]
            
        # Ensure channel count is 3 (RGB)
        if img_arr.shape[-1] != 3:
            print(f"[FATAL ERROR] Image does not have 3 channels! It has {img_arr.shape[-1]}.")
            print("Is this a Mask file (labels)? Prediction requires an RGB image.")
            return

        # --- Diagnosis Step 2: Check Value Range ---
        v_min, v_max, v_mean = img_arr.min(), img_arr.max(), img_arr.mean()
        print(f"Stats: Min={v_min:.4f}, Max={v_max:.4f}, Mean={v_mean:.4f}")
        
        # --- Diagnosis Step 3: Check for Empty/Dark Images ---
        if img_arr.mean() < 0.05:
            print("[WARN] Image is extremely dark (Mean < 0.05). Output might be unreliable.")
        
        # Prepare Input Tensor
        img_input = np.expand_dims(img_arr, axis=0) # Add batch dim
        
        # --- Inference ---
        print("\n[Running Inference...]")
        
        # Standard Model Prediction
        pred_std_raw = self.model_std.predict(img_input, verbose=0)
        pred_std_class = np.argmax(pred_std_raw[0], axis=-1)
        # Statistics
        unique, counts = np.unique(pred_std_class, return_counts=True)
        print(f"Standard Model Predicted Classes: {dict(zip(unique, counts))}")
        
        # Plus2 Model Prediction
        pred_plus2_raw = self.model_plus2.predict(img_input, verbose=0)
        # Handle multiple outputs (Deep Supervision)
        if isinstance(pred_plus2_raw, list): 
            main_pred = pred_plus2_raw[0]
        else: 
            main_pred = pred_plus2_raw
        pred_plus2_class = np.argmax(main_pred[0], axis=-1)
        # Statistics
        unique, counts = np.unique(pred_plus2_class, return_counts=True)
        print(f"Plus2 Model Predicted Classes:    {dict(zip(unique, counts))}")
        
        self.visualize_comparison(img_arr, pred_std_class, pred_plus2_class, save_path, gt_mask)
        
    def decode_mask(self, mask):
        """ Convert class indices to RGB mask based on color map. """
        r = np.zeros_like(mask).astype(np.uint8)
        g = np.zeros_like(mask).astype(np.uint8)
        b = np.zeros_like(mask).astype(np.uint8)
        for class_idx in range(0, self.num_classes):
            idx = (mask == class_idx)
            r[idx] = self.colors[class_idx, 0]
            g[idx] = self.colors[class_idx, 1]
            b[idx] = self.colors[class_idx, 2]
        return np.stack([r, g, b], axis=2)

    def visualize_comparison(self, original_img, pred_std, pred_plus2, save_path=None, gt_mask=None):
        """ Visualize Original Image, Ground Truth Mask, Standard Prediction, and Plus2 Prediction side-by-side. """
        mask_std_rgb = self.decode_mask(pred_std)
        mask_plus2_rgb = self.decode_mask(pred_plus2)
        
        # Convert float image back to uint8 for display
        if original_img.max() <= 1.0:
            original_img_disp = (original_img * 255).astype(np.uint8)
        else:
            original_img_disp = original_img.astype(np.uint8)
        
        # Determine number of subplots based on whether ground truth is available
        num_plots = 4 if gt_mask is not None else 3
        plt.figure(figsize=(5 * num_plots, 5))
        
        # Plot 1: Input
        plt.subplot(1, num_plots, 1)
        plt.title("Input (Rescaled)", fontsize=12, fontweight='bold')
        plt.imshow(original_img_disp)
        plt.axis('off')
        
        # Plot 2: Ground Truth (if available)
        if gt_mask is not None:
            mask_gt_rgb = self.decode_mask(gt_mask)
            plt.subplot(1, num_plots, 2)
            plt.title("Ground Truth", fontsize=12, fontweight='bold')
            plt.imshow(mask_gt_rgb)
            plt.axis('off')
            plot_offset = 1
        else:
            plot_offset = 0
        
        # Plot 3: Standard Result
        plt.subplot(1, num_plots, 2 + plot_offset)
        plt.title("Standard Model", fontsize=12, fontweight='bold')
        plt.imshow(mask_std_rgb)
        plt.axis('off')
        
        # Plot 4: Plus2 Result
        plt.subplot(1, num_plots, 3 + plot_offset)
        plt.title("Plus2 Model", fontsize=12, fontweight='bold')
        plt.imshow(mask_plus2_rgb)
        plt.axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n[DONE] Saved visualization to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--test_dir", type=str, default=TEST_IMAGES_DIR, 
                        help="Directory containing test images")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, 
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all .npy files in test directory
    test_images = sorted(glob(os.path.join(args.test_dir, "*.npy")))
    
    if len(test_images) == 0:
        print(f"[ERROR] No .npy files found in {args.test_dir}")
        sys.exit(1)
    
    print(f"\n[CONFIG] Test directory: {args.test_dir}")
    print(f"[CONFIG] Output directory: {args.output_dir}")
    print(f"[CONFIG] Found {len(test_images)} test images\n")
    
    # Initialize predictor
    predictor = LandCoverPredictor(DEFAULT_STD_WEIGHTS, DEFAULT_PLUS2_WEIGHTS)
    
    # Process each test image
    for idx, image_path in enumerate(test_images, 1):
        # Extract filename without extension
        filename = os.path.basename(image_path).replace('.npy', '')
        output_path = os.path.join(args.output_dir, f"{filename}_result.png")
        
        print(f"\n{'='*60}")
        print(f"Processing [{idx}/{len(test_images)}]: {filename}")
        print(f"{'='*60}")
        
        try:
            predictor.predict_single_npy(image_path, save_path=output_path)
        except Exception as e:
            print(f"[ERROR] Failed to process {filename}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total images processed: {len(test_images)}")
    print(f"Results saved to: {args.output_dir}")