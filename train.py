# import argparse
# import logging
# import os
# import sys

# import numpy as np
# import torch
# import torch.nn as nn
# from torch import optim
# from torch.backends import cudnn
# from tqdm import tqdm
# import matplotlib.pyplot as plt  # Added for plotting

# from utils.eval import eval_net
# # from unet import UNet
# from mobilenet.UNet_MobileNet import UNet

# from torch.utils.tensorboard import SummaryWriter
# from utils.dataset import BasicDataset
# from torch.utils.data import DataLoader, random_split

# # Input image and label paths
# dir_img = r"C:\Users\tygsh\OneDrive\Desktop\KIE4002_FYP\Training_Dataset\Bacterial Leaf Blight\Training_Ori"
# dir_mask = r"C:\Users\tygsh\OneDrive\Desktop\KIE4002_FYP\Training_Dataset\Bacterial Leaf Blight\Training_GT"
# dir_checkpoint = r"C:\Users\tygsh\OneDrive\Desktop\KIE4002_FYP\Code\UNet-MobileNet-Pytorch\checkpoints"


# def plot_and_save_graphs(loss_hist, val_hist, lr_hist, output_dir=r"C:\Users\tygsh\OneDrive\Desktop\KIE4002_FYP\Code\UNet-MobileNet-Pytorch\graphs"):
#     """Plots and saves training graphs."""
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     # 1. Plot Training Loss
#     if loss_hist:
#         plt.figure(figsize=(10, 5))
#         steps, losses = zip(*loss_hist)
#         plt.plot(steps, losses, label='Training Loss')
#         plt.title('Training Loss over Steps')
#         plt.xlabel('Global Step')
#         plt.ylabel('Loss')
#         plt.legend()
#         plt.grid(True)
#         plt.savefig(os.path.join(output_dir, 'training_loss.png'))
#         plt.close()

#     # 2. Plot Validation Score (Dice or CrossEntropy)
#     if val_hist:
#         plt.figure(figsize=(10, 5))
#         steps, scores = zip(*val_hist)
#         plt.plot(steps, scores, label='Validation Score', color='orange')
#         plt.title('Validation Score over Steps')
#         plt.xlabel('Global Step')
#         plt.ylabel('Score')
#         plt.legend()
#         plt.grid(True)
#         plt.savefig(os.path.join(output_dir, 'validation_score.png'))
#         plt.close()

#     # 3. Plot Learning Rate
#     if lr_hist:
#         plt.figure(figsize=(10, 5))
#         steps, lrs = zip(*lr_hist)
#         plt.plot(steps, lrs, label='Learning Rate', color='green')
#         plt.title('Learning Rate over Steps')
#         plt.xlabel('Global Step')
#         plt.ylabel('Learning Rate')
#         plt.legend()
#         plt.grid(True)
#         plt.savefig(os.path.join(output_dir, 'learning_rate.png'))
#         plt.close()
    
#     logging.info(f"Graphs saved to {output_dir}/")


# def train_net(net,
#               device,
#               epochs=5,
#               batch_size=1,
#               lr=0.001,
#               val_percent=0.1,
#               save_cp=True,
#               img_scale=0.5):

#     dataset = BasicDataset(dir_img, dir_mask, img_scale)
#     n_val = int(len(dataset) * val_percent)
#     n_train = len(dataset) - n_val
#     train, val = random_split(dataset, [n_train, n_val])
#     train_loader = DataLoader(
#         train, batch_size=batch_size, shuffle=True, num_workers=0)
#     val_loader = DataLoader(val, batch_size=batch_size,
#                             shuffle=False, num_workers=0, drop_last=True)

#     writer = SummaryWriter(
#         comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
#     global_step = 0

#     logging.info(f'''Starting training:
#         Epochs:          {epochs}
#         Batch size:      {batch_size}
#         Learning rate:   {lr}
#         Training size:   {n_train}
#         Validation size: {n_val}
#         Checkpoints:     {save_cp}
#         Device:          {device.type}
#         Images scaling:  {img_scale}
#     ''')
    
#     # --- Data collection lists ---
#     loss_history = []
#     val_score_history = []
#     lr_history = []
#     # -----------------------------

#     # Define optimizer
#     optimizer = optim.RMSprop(net.parameters(), lr=lr,
#                               weight_decay=1e-8, momentum=0.9)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, 'min' if net.num_classes > 1 else 'max', patience=2)

#     # Define loss function
#     if net.num_classes > 1:
#         criterion = nn.CrossEntropyLoss()
#     else:
#         criterion = nn.BCEWithLogitsLoss()

#     # Start training
#     try:
#         for epoch in range(epochs):
#             net.train()

#             epoch_loss = 0
#             with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
#                 for batch in train_loader:
#                     imgs = batch['image']
#                     true_masks = batch['mask']
#                     assert imgs.shape[1] == net.n_channels, \
#                         f'Network has been defined with {net.n_channels} input channels, ' \
#                         f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
#                         'the images are loaded correctly.'

#                     imgs = imgs.to(device=device, dtype=torch.float32)
#                     mask_type = torch.float32 if net.num_classes == 1 else torch.long
#                     true_masks = true_masks.to(device=device, dtype=mask_type)
#                     masks_pred = net(imgs)
#                     loss = criterion(masks_pred, true_masks)
#                     epoch_loss += loss.item()
#                     writer.add_scalar('Loss/train', loss.item(), global_step)

#                     # --- Record Training Loss ---
#                     loss_history.append((global_step, loss.item()))
#                     # ----------------------------

#                     pbar.set_postfix(**{'loss (batch)': loss.item()})

#                     optimizer.zero_grad()
#                     loss.backward()
#                     nn.utils.clip_grad_value_(net.parameters(), 0.1)
#                     optimizer.step()

#                     pbar.update(imgs.shape[0])
#                     global_step += 1
#                     if global_step % (n_train // (10 * batch_size)) == 0:
#                         for tag, value in net.named_parameters():
#                             tag = tag.replace('.', '/')
#                             writer.add_histogram(
#                                 'weights/' + tag, value.data.cpu().numpy(), global_step)
#                             # writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
#                         val_score = eval_net(net, val_loader, device)
#                         scheduler.step(val_score)
#                         writer.add_scalar(
#                             'learning_rate', optimizer.param_groups[0]['lr'], global_step)

#                         # --- Record Validation Score and LR ---
#                         val_score_history.append((global_step, val_score))
#                         lr_history.append((global_step, optimizer.param_groups[0]['lr']))
#                         # --------------------------------------

#                         if net.num_classes > 1:
#                             logging.info(
#                                 'Validation cross entropy: {}'.format(val_score))
#                             writer.add_scalar('Loss/test', val_score, global_step)
#                         else:
#                             logging.info(
#                                 'Validation Dice Coeff: {}'.format(val_score))
#                             writer.add_scalar('Dice/test', val_score, global_step)

#                         writer.add_images('images', imgs, global_step)
#                         if net.num_classes == 1:
#                             writer.add_images(
#                                 'masks/true', true_masks, global_step)
#                             writer.add_images(
#                                 'masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

#             # Save model
#             if save_cp and (not epoch % 10):
#                 try:
#                     os.mkdir(dir_checkpoint)
#                     logging.info('Created checkpoint directory')
#                 except OSError:
#                     pass
#                 torch.save(net.state_dict(),
#                            os.path.join(dir_checkpoint, f'MobileNet_UNet_epoch{epoch + 1}.pt'))
#                 logging.info(f'Checkpoint {epoch + 1} saved !')

#     except KeyboardInterrupt:
#         torch.save(net.state_dict(), 'INTERRUPTED.pt')
#         logging.info('Saved interrupt')
#         raise
    
#     finally:
#         writer.close()
#         # --- Plot Graphs on Exit ---
#         plot_and_save_graphs(loss_history, val_score_history, lr_history)
#         # ---------------------------


# def get_args():
#     parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
#                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument('-e', '--epochs', metavar='E', type=int, default=100,
#                         help='Number of epochs', dest='epochs')
#     parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
#                         help='Batch size', dest='batchsize')
#     parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
#                         help='Learning rate', dest='lr')
#     parser.add_argument('-f', '--load', dest='load', type=str, default='',
#                         help='Load model from a .pth file')
#     parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
#                         help='Downscaling factor of the images')
#     parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
#                         help='Percent of the data that is used as validation (0-100)')

#     return parser.parse_args()


# if __name__ == '__main__':
#     logging.basicConfig(level=logging.INFO,
#                         format='%(levelname)s: %(message)s')
#     args = get_args()

#     # Determine whether to use GPU
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logging.info(f'Using device {device}')

#     # Import network model
#     # n_channels=3 for RGB images
#     # num_classes logic: 1 for 1 class+background, or 2 classes. N for >2 classes.
#     net = UNet(n_channels=3, num_classes=1)

#     logging.info(f'Network:\n'
#                  f'\t{net.n_channels} input channels\n'
#                  f'\t{net.num_classes} output channels (classes)\n')

#     # Load pretrained weights if provided
#     if args.load:
#         model_dict = net.state_dict()
#         model_path = args.load
#         pretrained_dict = torch.load(model_path, map_location=device)
#         # Filter out unnecessary layers
#         pretrained_dict = {k: v for k,
#                            v in pretrained_dict.items() if k in model_dict}
#         # Update current network dictionary
#         model_dict.update(pretrained_dict)
#         net.load_state_dict(model_dict)
#         logging.info(f'Model loaded from {args.load}')

#     net.to(device=device)

#     # faster convolutions, but more memory
#     cudnn.benchmark = True

#     try:
#         train_net(net=net,
#                   epochs=args.epochs,
#                   batch_size=args.batchsize,
#                   lr=args.lr,
#                   device=device,
#                   img_scale=args.scale,
#                   val_percent=args.val / 100)
#     except KeyboardInterrupt:
#         pass  # Handled inside train_net now

import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.backends import cudnn
from tqdm import tqdm

from utils.eval import eval_net
# from unet import UNet
from mobilenet.UNet_MobileNet import UNet
from utils.dice_loss import dice_coeff  # <--- Added this import

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split

# Input image and label paths
dir_img = 'data/liver/liver/train'
dir_mask = 'data/liver/liver/masks'
dir_checkpoint = 'data/liver/checkpoints'


def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5):

    dataset = BasicDataset(dir_img, dir_mask, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val, batch_size=batch_size,
                            shuffle=False, num_workers=0, drop_last=True)

    writer = SummaryWriter(
        comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')
    
    optimizer = optim.RMSprop(net.parameters(), lr=lr,
                              weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min' if net.num_classes > 1 else 'max', patience=2)

    if net.num_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    # Initialize validation metrics for display
    val_loss = 0.0
    val_dice = 0.0

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        # Increased bar width for better visibility
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img', ncols=160) as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.num_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)
                
                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                # --- Calculate Train Dice for Display ---
                train_dice = 0.0
                if net.num_classes == 1:
                    pred_binary = (torch.sigmoid(masks_pred) > 0.5).float()
                    train_dice = dice_coeff(pred_binary, true_masks).item()
                # ----------------------------------------

                # --- Update Progress Bar ---
                pbar.set_postfix(**{
                    'T_Loss': f'{loss.item():.4f}',
                    'T_Dice': f'{train_dice:.4f}',
                    'V_Loss': f'{val_loss:.4f}',
                    'V_Dice': f'{val_dice:.4f}',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })
                # ---------------------------

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                
                # Validation Step
                if global_step % (n_train // (10 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)

                    # Get both Loss and Dice from updated eval_net
                    val_loss, val_dice = eval_net(net, val_loader, device)
                    
                    # Scheduler uses Dice (maximize) or Loss (minimize)
                    # For 1 class, we usually want to maximize Dice
                    scheduler.step(val_dice)

                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
                    writer.add_scalar('Loss/test', val_loss, global_step)
                    writer.add_scalar('Dice/test', val_dice, global_step)

                    logging.info(f' Validation Dice: {val_dice:.4f}, Validation Loss: {val_loss:.4f}')

                    writer.add_images('images', imgs, global_step)
                    if net.num_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

        if save_cp and (not epoch % 10):
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       os.path.join(dir_checkpoint, f'MobileNet_UNet_epoch{epoch + 1}.pt'))
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=100,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default='',
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = UNet(n_channels=3, num_classes=1)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.num_classes} output channels (classes)\n')

    if args.load:
        model_dict = net.state_dict()
        model_path = args.load
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pt')
        logging.info('Saved interrupt')