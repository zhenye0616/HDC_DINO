import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from util.misc import nested_tensor_from_tensor_list



class DINOBackbone(nn.Module):
    def __init__(self, dino_model):
        super().__init__()
        self.backbone = dino_model.backbone
        self.input_proj = dino_model.input_proj
    def forward(self, samples):
        features, poss = self.backbone(samples)
        # Use the highest-resolution feature level (or combine them as needed)
        src, mask = features[-1].decompose()
        src_proj = self.input_proj[-1](src)
        return src_proj, mask, poss

# # Usage:
# full_model, criterion, postprocessors = build_model_main(args)
# backbone_only = DINOBackbone(full_model)
# src_proj, mask, poss = backbone_only(samples)
# print("Backbone output shape:", src_proj.shape)


def check_activation_stats(encoder, model, dataloader, device, num_batches=1):
    #encoder.eval()  # set encoder to eval mode
    all_means = []
    all_vars = []
    all_activations = []

    with torch.no_grad():
        batch_count = 0
        for batch in dataloader:
            images, _ = batch
            # If images is a NestedTensor, extract the tensor
            if hasattr(images, 'tensors'):
                images = images.tensors
            images = images.to(device)
            
            # ---- Extract backbone features first ----
            # The backbone returns a list of feature maps; we choose one level (e.g., the last)
            backbone = DINOBackbone(model)
            features, poss = backbone(images)
            # Pick a feature map; for example, use the last level
            feat = features[-1]  # This is a NestedTensor
            src, mask = feat.decompose()  # src shape: [B, 256, H, W]
            
            # Project the features to match the transformer's expected dimension.
            # Use the corresponding projection layer from model.input_proj.
            src_proj = model.input_proj[-1](src)  # shape remains [B, 256, H, W]
            
            # Flatten the spatial dimensions into tokens:
            B, C, H, W = src_proj.shape
            # Reshape to [B, T, C] where T = H*W
            tokens = src_proj.view(B, C, H * W).permute(0, 2, 1)
            # Now tokens has shape [B, T, 256], which is what the encoder expects.
            
            # ---- Pass tokens through the HD encoder ----
            activations = encoder(tokens)
            # activations shape: [B, T, dim] (e.g., [B, T, 4000])
            
            # Compute statistics across the tokens dimension
            batch_mean = activations.mean(dim=0)  # shape: [T, dim] -> average over batch
            batch_var = activations.var(dim=0)
            
            all_means.append(batch_mean.cpu().numpy())
            all_vars.append(batch_var.cpu().numpy())
            all_activations.append(activations.cpu().numpy())
            
            batch_count += 1
            if batch_count >= num_batches:
                break

    # Combine statistics from all batches
    all_means = np.concatenate([m.reshape(1, -1) for m in all_means], axis=0)
    all_vars = np.concatenate([v.reshape(1, -1) for v in all_vars], axis=0)
    all_activations = np.concatenate(all_activations, axis=0)

    overall_mean = all_means.mean(axis=0)
    overall_var = all_vars.mean(axis=0)

    print("Overall Activation Mean (per feature dimension):", overall_mean)
    print("Mean of Means:", overall_mean.mean())
    print("Overall Activation Variance (per feature dimension):", overall_var)
    print("Mean of Variances:", overall_var.mean())

    # Plot histogram of activation values
    plt.hist(all_activations.ravel(), bins=50)
    plt.title("Histogram of HD Encoder Activations")
    plt.xlabel("Activation Value")
    plt.ylabel("Frequency")
    plt.show()

# Example usage:
# Here, 'model' is your full DINO model (returned from build_model_main),
# and you are checking the activation statistics of the encoder in your HD head.
# Make sure to pass the full model to access the backbone and input_proj.
# check_activation_stats(model.class_embed[0].encoder, model, dataloader, device)
