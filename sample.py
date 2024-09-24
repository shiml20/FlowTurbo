import torch
import time
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from models_assemble import FlowTurboAssemble

from PIL import Image
from IPython.display import display
torch.set_grad_enabled(False)

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("GPU not found. Using CPU instead.")


cfg_scale = 4
class_labels = [207, 980, 387, 974, 88, 972, 928, 279]
image_size = "256"

refiner_ckpt = 'refiner_ckpt'
predictor_ckpt = 'predictor_ckpt'
vae_ckpt = "stabilityai/sd-vae-ft-ema"
latent_size = int(image_size) // 8

vae = AutoencoderKL.from_pretrained(vae_ckpt).to(device)


seed = 0
torch.manual_seed(seed)
samples_per_row = 4
# Create sampling noise:
n = len(class_labels)
z = torch.randn(n, 4, latent_size, latent_size, device=device)
y = torch.tensor(class_labels, device=device)

# Setup classifier-free guidance:
z = torch.cat([z, z], 0)
y_null = torch.tensor([1000] * n, device=device)
y = torch.cat([y, y_null], 0)
model_kwargs = dict(y=y, cfg_scale=cfg_scale)

sample_config = [{'N_H': 16, 'N_P': 18, 'N_R': 10, 'SAC': True}]

FlowTurbo = FlowTurboAssemble(predictor_ckpt=predictor_ckpt, refiner_ckpt=refiner_ckpt, vae_ckpt=vae_model, **sample_config[0])
FlowTurbo.eval()
FlowTurbo.to(device)


with torch.autocast(device_type="cuda"):
    samples = FlowTurbo(z, **model_kwargs)
    image_path =f'sample.png'
    save_image(samples, image_path, nrow=int(samples_per_row), 
            normalize=True, value_range=(-1, 1))
    print(f"Images are saved in {image_path}")

