import torch
import yaml
from torch.utils.data import DataLoader
from pathlib import Path
from ecoperceiver.dataset import EcoPerceiverLoaderConfig
from carboninference.core import ERA5Dataset
from ecoperceiver.components import EcoPerceiverConfig
from ecoperceiver.model import EcoPerceiver

config_path = Path('./config.yml')
checkpoint_path = Path('../runs/test_run_3e-06_ws_l128_f12_e32_c32_o0.3_wcswcswcswcsssss_CC/seed_0/last.pth')
data_path = Path('./data')

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

print("Configuration loaded:")
print(f"Model targets: {config['model']['targets']}")
print(f"Context length: {config['model']['context_length']}")
print(f"Latent space dim: {config['model']['latent_space_dim']}")
print(f"Checkpoint path: {checkpoint_path}")
print(f"Checkpoint exists: {checkpoint_path.exists()}")

model_config = EcoPerceiverConfig(
    targets=tuple(config['model']['targets']),
    latent_space_dim=config['model']['latent_space_dim'],
    num_frequencies=config['model']['num_frequencies'],
    input_embedding_dim=config['model']['input_embedding_dim'],
    context_length=config['model']['context_length'],
    obs_dropout=config['model']['obs_dropout'],
    weight_sharing=config['model']['weight_sharing'],
    layers=config['model']['layers'],
    pretrained_path=config['model']['pretrained_path']
)

model = EcoPerceiver(model_config)
print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

if checkpoint_path.exists():
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=torch.device('cpu'))
    
    model.load_state_dict(checkpoint['model'])
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    print(f"Model moved to {device} and set to evaluation mode")
else:
    print(f"Checkpoint not found at {checkpoint_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Model moved to {device} (no checkpoint loaded)")

# Create dataset and dataloader
dataset_config = EcoPerceiverLoaderConfig(**config['dataset'])
dataset = ERA5Dataset(data_path, config=dataset_config)

# Create dataloader
dataloader = DataLoader(
    dataset, 
    batch_size=64,  
    shuffle=False,  
    num_workers=8, 
    pin_memory=True, 
    collate_fn=dataset.collate_fn
)

print(f"Dataset created with {len(dataset)} samples")

# Test model inference
print("Testing model inference...")
with torch.no_grad():
    batch = next(iter(dataloader))
    if hasattr(batch, "to"):
        batch = batch.to(device)

    res = model(batch) # flux_labels, predictions, loss

    yhat = res.predictions
    print(f"Pred shape: {yhat.shape}")
    print(f"Loss: {float(res.loss.mean().item()):.4f}")
