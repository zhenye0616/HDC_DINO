import os, sys
#sys.path.append('/home/biaslab/Zhen/HDC_DINO')
import argparse
import pickle
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import json
import time
from util import misc
import util.misc as utils
from Demo.finetune import check_activation_stats, DINOBackbone
import math
import torch.nn as nn
from util import misc
best_acc = 0.0

def collate_with_orig(batch):
    images, targets = utils.collate_fn(batch)
    for t in targets:
        # if your pipeline put the image’s original H×W under "size", just copy it
        t["orig_size"] = t.get("size")
    return images, targets

def print_grad(grad):
    print("Gradient norm:", grad.norm().item())

def log_head_gradients(model):
    for i, head in enumerate(model.class_embed):
        if head.encoder.weight.grad is not None:
            print(f"Head {i} encoder.weight grad norm: {head.encoder.weight.grad.norm().item():.4f}")
        if head.encoder.bias is not None and head.encoder.bias.grad is not None:
            print(f"Head {i} encoder.bias grad norm: {head.encoder.bias.grad.norm().item():.4f}")
        if head.model.bias.grad is not None:
            print(f"Head {i} model.bias grad norm: {head.model.bias.grad.norm().item():.4f}")


def train_hd(model, criterion, postprocessors, dataloader, test_loader, num_epochs, device, args):
    from main import build_model_main, train_one_epoch, evaluate
    global best_acc  
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("hd_ckpt_5000", exist_ok=True)  
    

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
   
    # Unfreeze the classification head parameters (assumed to be in model.class_embed)
    for head in model.class_embed:
        head.encoder = head.encoder.to(args.device)
        head.model = head.model.to(args.device)
        #import pdb;pdb.set_trace()
        for param in head.encoder.parameters():
            param.requires_grad = False #True
        #head.encoder.basis.requires_grad_(True)
        #head.encoder.base.requires_grad_(True)
        head.model.requires_grad_(True)

    # Unfreeze the regression head parameters (assumed to be in model.bbox_embed)
    for layer in model.bbox_embed:
        for param in layer.parameters():
            param.requires_grad = True

    # Debug print to check trainable parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.requires_grad}")
            
    model.to(device)

    # Set up the optimizer to update parameters for both the classification and regression heads
    optimizer = torch.optim.Adam(
        list(model.class_embed.parameters()) + list(model.bbox_embed.parameters()),
        lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # Update log header to include test loss
    log_file = open(os.path.join(log_dir, "training_log_hd_5000.txt"), "w")
    log_file = open(os.path.join(log_dir, "training_log_hd_5000.txt"), "w")
    log_file.write(
        "Epoch, SamplesSeen, TrainLoss, LR, EpochTime(s), CumulativeTime(s)\n"
    )
    log_file.flush()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        # Train for one epoch
        #check_activation_stats(model.class_embed[0].encoder, model, dataloader, args.device)
        #import pdb;pdb.set_trace()
        epoch_stats = train_one_epoch(
            model, criterion, dataloader, optimizer,
            device, epoch, max_norm=0, wo_class_error=False,
            lr_scheduler=scheduler, args=args, logger=None, ema_m=None
        )

        epoch_time = time.time() - epoch_start_time
        cumulative_time += epoch_time
        avg_train_loss = epoch_stats.get("loss", 0.0)
        lr = optimizer.param_groups[0]["lr"]

        # Evaluate on test set
        if args.eval_only:
            pass #add this later
        
        samples_seen = (epoch + 1) * len(dataloader) * args.batch_size
        # Log epoch statistics
        #log_data = f"{epoch}, {avg_train_loss:.4f},  {lr:.6f}, {epoch_time:.2f}\n"
        log_data = (
                f"{epoch+1},"
                f"{samples_seen},"
                f"{avg_train_loss:.4f},"
                f"{val_loss:.4f},"
                f"{val_mAP50:.4f},"
                f"{lr:.6f},"
                f"{epoch_time:.2f},"
                f"{cumulative_time:.2f}\n"
        )
        log_file.write(log_data)
        log_file.flush()
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f} s, avg train loss: {avg_train_loss:.4f},  lr: {lr:.6f}")

        # Save checkpoints for regression head and full model
        torch.save(model.bbox_embed.state_dict(), f'hd_ckpt_100/regression_head_epoch_{epoch}.pth')
        torch.save(model.state_dict(), f"hd_ckpt_100/model_epoch_{epoch}.pth")

        # Save classification head checkpoint
        print(f"Saving checkpoint for Epoch {epoch}")
        torch.save(model.class_embed.state_dict(), f'hd_ckpt_100/hd_classification_head_epoch_{epoch}.pth')
    
    log_file.close()

# Use the original DINO dataset loader and collate function
def load_dino_data(args):
    from datasets import build_dataset  # Use DINO's dataset builder
    # Build train and validation datasets using DINO's configuration
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    
    # Use the original collate function from DINO's utils
    data_loader_train = DataLoader(
        dataset_train, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=utils.collate_fn, 
        num_workers = args.num_workers,  # Use the official collate_fn
    )
    data_loader_val = DataLoader(
        dataset_val, 
        batch_size=args.batch_size, 
        shuffle=False,
        collate_fn=collate_with_orig,
        num_workers = args.num_workers,
    )
    try:
        coco_gt = dataset_val.coco        # torchvision COCO‐style dataset
    except AttributeError:
        coco_gt = dataset_val.coco_api
    return data_loader_train, data_loader_val, coco_gt

def main(args):
    from main import build_model_main, train_one_epoch, evaluate
    model, criterion, postprocessors = build_model_main(args)
    # Load checkpoint and filter out keys for modified heads
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    state_dict = checkpoint['model']  # Assuming checkpoint is the state dict
    filtered_state_dict = {
        k: v for k, v in state_dict.items() 
        if "class_embed" not in k and "bbox_embed" not in k
    }
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)
    model.to(args.device)
    #base_ds = None

    # Load coco names if needed
    with open('/home/biaslab/Zhen/HDC_DINO/util/coco_id2name.json') as f:
        id2name = json.load(f)
        id2name = {int(k):v for k,v in id2name.items()}

    # Use the DINO dataset loader to load training and validation data
    dataloader, test_loader, base_ds = load_dino_data(args)
    #check backbone output
    # for batch in dataloader:
    #     images, _ = batch
    #     backbone = DINOBackbone(model)
    #     src_proj, mask, poss = backbone(images.to(args.device))
    #     print("Backbone output shape:", src_proj.shape)
    HDC = False
    if args.eval_only:
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location=args.device)
            state_dict = checkpoint['model']
            model.load_state_dict(state_dict, strict=False)
            if HDC :
                filtered_state_dict = {k: v for k, v in state_dict.items() if "class_embed" not in k}
                model.load_state_dict(filtered_state_dict, strict=False)
                custom_class_embed = torch.load("/home/biaslab/Zhen/DINO/Demo/hd_ckpt_100/hd_classification_head_epoch_11.pth", map_location=args.device)
                custom_bbox_embed = torch.load("/home/biaslab/Zhen/DINO/Demo/hd_ckpt_100/regression_head_epoch_11.pth", map_location=args.device)
                model.class_embed.load_state_dict(custom_class_embed)
                model.bbox_embed.load_state_dict(custom_bbox_embed)
            print("Loaded checkpoint")
        stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, test_loader, base_ds, 
            args.device, args.output_dir, args=args
        )
        print("Evaluation stats:", stats)
    else:
        train_hd(model, criterion, postprocessors, dataloader, test_loader,
                  num_epochs=args.num_epochs, device=args.device, args=args)
    
def HD_init(hd_model, prior_prob=0.01):
    # Initialize the linear encoder with Xavier uniform initialization.
    nn.init.xavier_uniform_(hd_model.encoder.weight)
    if hd_model.encoder.bias is not None:
        nn.init.constant_(hd_model.encoder.bias, 0)
        
    # Initialize the final classification layer (the "model" attribute) with Xavier uniform.
    nn.init.xavier_uniform_(hd_model.model.weight)
    
    # Compute the bias value so that the initial logits correspond to a low probability.
    bias_value = -math.log((1 - prior_prob) / prior_prob)
    hd_model.model.bias.data = torch.ones(hd_model.classes) * bias_value
    print('HD_init')


if __name__ == "__main__":
    import argparse
    from util.slconfig import SLConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_only", action="store_true", help="Run evaluation only without training")
    parser.add_argument("--checkpoint", type=str, default="/home/biaslab/Zhen/HDC_DINO/checkpoint/checkpoint0033_4scale.pth", help="Path to the checkpoint file")
    parser.add_argument('--num_epochs', type=int, default=12)
    parser.add_argument('--num_classes', type=int, default=91)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--amp', action='store_true',help="Train with mixed precision")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--config', type=str, default="/home/biaslab/Zhen/HDC_DINO/config/DINO/DINO_4scale.py")
    parser.add_argument('--output_dir', default='',help='path where to save, empty for no saving')
    parser.add_argument('--dataset_file', default='coco', help="Name of the dataset, e.g. 'coco'")
    parser.add_argument('--coco_path', type=str, default='/mnt/Data_1/coco')
    parser.add_argument('--fix_size', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument("--save_results",action="store_true",help="Whether to dump JSON results during evaluation")
    cli_args = parser.parse_args()
   
    file_args = SLConfig.fromfile(cli_args.config)
    cli_args_dict = vars(cli_args)
    for key, value in cli_args_dict.items():
        file_args[key] = value
    file_args.device = 'cuda'

    main(file_args)
