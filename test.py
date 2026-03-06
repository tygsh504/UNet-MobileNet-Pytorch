import argparse
import logging
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# Import your model and dataset utils
from mobilenet.UNet_MobileNet import UNet
from utils.dataset import BasicDataset

# ==========================================
# 1. METRICS FUNCTIONS
# ==========================================
def calculate_metrics(pred_mask, true_mask):
    """
    Calculates metrics for a single image (numpy arrays).
    Expects binary inputs (0 or 1).
    """
    # Flatten
    y_pred = pred_mask.flatten()
    y_true = true_mask.flatten()
    
    # Constants
    smooth = 1e-6
    
    # True Positives, False Positives, False Negatives, True Negatives
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    
    # 1. Dice Coefficient (F1 Score equivalent for binary)
    dice = (2. * TP + smooth) / (2. * TP + FP + FN + smooth)
    
    # 2. IoU (Jaccard Index)
    iou = (TP + smooth) / (TP + FP + FN + smooth)
    
    # 3. Precision
    precision = (TP + smooth) / (TP + FP + smooth)
    
    # 4. Recall (Sensitivity)
    recall = (TP + smooth) / (TP + FN + smooth)
    
    # 5. Accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    # 6. F1 Score (Calculated from Precision/Recall)
    f1 = 2 * (precision * recall) / (precision + recall + smooth)
    
    return dice, iou, precision, recall, accuracy, f1

# ==========================================
# 2. FLOPs COUNTER (Hook Based)
# ==========================================
def count_flops_and_params(model, input_size=(1, 3, 256, 256), device='cpu'):
    """
    Estimates FLOPs by registering hooks on Conv2d and Linear layers.
    Note: This is an estimation.
    """
    flops = 0
    
    def conv_hook(self, input, output):
        nonlocal flops
        batch_size, input_channels, input_height, input_width = input[0].shape
        output_channels, output_height, output_width = output.shape[1:]
        
        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels // self.groups)
        bias_ops = 1 if self.bias is not None else 0
        
        # ops per output element
        params = kernel_ops + bias_ops
        flops += batch_size * params * output_height * output_width * output_channels

    def linear_hook(self, input, output):
        nonlocal flops
        batch_size = input[0].shape[0]
        weight_ops = self.weight.nelement()
        bias_ops = self.bias.nelement() if self.bias is not None else 0
        flops += batch_size * (weight_ops + bias_ops)

    hooks = []
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            hooks.append(layer.register_forward_hook(conv_hook))
        elif isinstance(layer, nn.Linear):
            hooks.append(layer.register_forward_hook(linear_hook))

    # Run dummy pass
    dummy_input = torch.rand(input_size).to(device)
    model(dummy_input)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    total_params = sum(p.numel() for p in model.parameters())
    return flops, total_params

# ==========================================
# 3. TESTING LOGIC
# ==========================================
def run_test(args):
    # Setup Paths
    output_folder = "testing_output"
    vis_folder = os.path.join(output_folder, "visualizations")
    os.makedirs(vis_folder, exist_ok=True)
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # 1. Load Model
    logging.info(f"Loading model from {args.model}")
    net = UNet(n_channels=3, num_classes=1)
    net.to(device)
    
    try:
        state_dict = torch.load(args.model, map_location=device)
        net.load_state_dict(state_dict)
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return

    net.eval()

    # 2. Calculate Params & FLOPs (Estimate)
    # Using the image scale to determine input size for FLOPs calculation
    # Assuming a standard roughly 512x512 input scaled by args.scale
    dummy_h = int(512 * args.scale) 
    dummy_w = int(512 * args.scale)
    logging.info(f"Calculating FLOPs with input shape (1, 3, {dummy_h}, {dummy_w})...")
    
    try:
        flops, params = count_flops_and_params(net, input_size=(1, 3, dummy_h, dummy_w), device=device)
        logging.info(f"Total Params: {params:,}")
        logging.info(f"Total FLOPs (Est): {flops:,}")
    except Exception as e:
        logging.warning(f"Could not calculate FLOPs: {e}")
        flops, params = 0, 0

    # 3. Prepare Data
    # finding images with common extensions
    img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    test_img_paths = []
    for ext in img_extensions:
        test_img_paths.extend(glob.glob(os.path.join(args.input_dir, "**", ext), recursive=True))
    
    test_img_paths = sorted(test_img_paths)
    
    if len(test_img_paths) == 0:
        logging.error(f"No images found in {args.input_dir}")
        return

    logging.info(f"Found {len(test_img_paths)} images. Starting evaluation...")
    
    results_list = []
    
    for i, img_path in enumerate(tqdm(test_img_paths, unit="img")):
        filename = os.path.basename(img_path)
        file_id = os.path.splitext(filename)[0]
        
        # Find corresponding mask
        # Assuming mask has same name as image (extensions might differ)
        mask_path = glob.glob(os.path.join(args.mask_dir, file_id + ".*"))
        
        if not mask_path:
            logging.warning(f"No mask found for {filename}, skipping...")
            continue
        mask_path = mask_path[0] # Take the first match

        # --- LOAD & PREPROCESS (Using BasicDataset logic) ---
        # 1. Open Images
        img_pil = Image.open(img_path)
        mask_pil = Image.open(mask_path)
        
        # 2. Preprocess (Resize & Normalize)
        # Note: preprocess returns numpy (C, H, W)
        img_np = BasicDataset.preprocess(img_pil, args.scale)
        mask_np = BasicDataset.preprocess(mask_pil, args.scale)
        
        # Convert to Tensor
        img_tensor = torch.from_numpy(img_np).type(torch.FloatTensor).unsqueeze(0).to(device) # (1, C, H, W)
        
        # Ground Truth Mask Processing
        # mask_np is (1, H, W) or (H, W). If standardized, it's (1, H, W).
        if mask_np.ndim == 3:
            mask_true = mask_np[0, :, :] # Take channel 0
        else:
            mask_true = mask_np
        
        # Binarize GT (just in case)
        mask_true = (mask_true > 0.5).astype(np.uint8)

        # --- INFERENCE ---
        with torch.no_grad():
            output = net(img_tensor)
            
            # Sigmoid for binary segmentation
            probs = torch.sigmoid(output)
            probs = probs.squeeze(0).squeeze(0).cpu().numpy() # (H, W)
            
            # Threshold
            mask_pred = (probs > args.mask_threshold).astype(np.uint8)

        # --- METRICS ---
        dice, iou_val, prec, recall, acc, f1 = calculate_metrics(mask_pred, mask_true)
        
        # --- SAVE VISUALIZATION ---
        # Denormalize image for display
        # BasicDataset preprocess divides by 255 if max > 1.
        # So inputs are [0, 1]. We just multiply by 255.
        if img_np.shape[0] == 3: # RGB
            img_vis = img_np.transpose((1, 2, 0)) # CHW -> HWC
        else: # Grayscale
            img_vis = img_np[0, :, :]
            
        img_vis = (img_vis * 255).astype(np.uint8)
        
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(img_vis)
        ax[0].set_title(f"Original: {filename}")
        ax[0].axis('off')
        
        ax[1].imshow(mask_true, cmap='gray')
        ax[1].set_title("Ground Truth")
        ax[1].axis('off')
        
        ax[2].imshow(mask_pred, cmap='gray')
        ax[2].set_title(f"Pred (Dice: {dice:.2f})")
        ax[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_folder, f"eval_{file_id}.png"))
        plt.close()

        # --- LOG RESULT ---
        results_list.append({
            "Filename": filename,
            "Dice": dice,
            "IoU": iou_val,
            "Precision": prec,
            "Recall": recall,
            "Accuracy": acc,
            "F1_Score": f1
        })

    # ==========================================
    # 4. SAVE TO EXCEL
    # ==========================================
    if not results_list:
        logging.error("No results generated. Check your paths.")
        return

    df = pd.DataFrame(results_list)
    avg_metrics = df.mean(numeric_only=True)
    
    summary_data = {
        "Metric": ["Average Dice", "Average IoU", "Average Precision", "Average Recall", "Average Accuracy", "Average F1", "Total Params", "FLOPs"],
        "Value": [
            avg_metrics["Dice"], 
            avg_metrics["IoU"], 
            avg_metrics["Precision"], 
            avg_metrics["Recall"], 
            avg_metrics["Accuracy"], 
            avg_metrics["F1_Score"],
            params,
            flops
        ]
    }
    
    excel_path = os.path.join(output_folder, "evaluation_results_pytorch.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)
        df.to_excel(writer, sheet_name="Detailed_Results", index=False)
        
    logging.info("------------------------------------------------")
    logging.info(f"Testing Complete. Results saved to: {excel_path}")
    logging.info(pd.DataFrame(summary_data))
    logging.info("------------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch UNet-MobileNet Testing Script')
    
    # Model Args
    parser.add_argument('--model', '-m', default='best_model.pt',
                        metavar='FILE', help="Path to the .pt model file")
    
    # Data Args
    # Set these defaults to your actual paths to avoid typing them every time
    default_imgs = r"C:\Users\tygsh\OneDrive\Desktop\KIE4002_FYP\Training_Dataset\Tungro\Infer_Ori"
    default_masks = r"C:\Users\tygsh\OneDrive\Desktop\KIE4002_FYP\Training_Dataset\Tungro\Infer_GT"
    
    parser.add_argument('--input-dir', '-i', metavar='INPUT', default=default_imgs,
                        help='Directory of input images')
    parser.add_argument('--mask-dir', '-g', metavar='MASK', default=default_masks,
                        help='Directory of ground truth masks')
    
    # Processing Args
    parser.add_argument('--scale', '-s', type=float, default=1,
                        help="Scale factor for the input images (default: 0.5 to match training)")
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help="Minimum probability value to consider a mask pixel white")

    args = parser.parse_args()
    run_test(args)