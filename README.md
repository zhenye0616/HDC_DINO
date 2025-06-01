# DINO + HDC Integration

This repository contains an extended version of DINO with support for various classification heads, including HDC. Follow the instructions below to set up the environment and run the model.

---

## 🔧 Setup Instructions

### 1. Create the Conda Environment

```bash
conda env create -f environment.yml
conda activate hdc_dino  # or whatever name you used
cd models/dino/ops
python setup.py build install

# Run unit tests (should output "All checking is True")
python test.py

cd ../../..

📦 Checkpoints & Training

Pretrained checkpoint is located at:
/checkpoint
Fine-tuning with HDC head is implemented in:
dino_hd.py
```

###⚠️ Important: Updating Classification Head

If you plan to change the classification head or load a different checkpoint, make sure to update the corresponding lines in dino.py, or the checkpoint will not match the model architecture.

### Example Snippet (from `dino.py`):

```python
# Choose one classification head
_class_embed = nn.Linear(hidden_dim, num_classes)               # Linear (default)
_class_embed = MLP(hidden_dim, hidden_dim, num_classes, 3)      # MLP
_class_embed = HD(num_classes, hidden_dim, dim=100).to('cuda')  # HDC

# BBox embed layer (always MLP)
_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

# Initialization (Linear)
prior_prob = 0.01
bias_value = -math.log((1 - prior_prob) / prior_prob)
_class_embed.bias.data = torch.ones(self.num_classes) * bias_value

# Initialization (BBox)
nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

# For MLP: uncomment and use this instead
# nn.init.constant_(_class_embed.layers[-1].bias, bias_value)

# For HDC: initialize using custom method
# HD_init(_class_embed)
