# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained SiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via evaluation/evaluator.py

"""

import torch
import torch.distributed as dist
from download import find_model
from transport import create_transport, Sampler
from diffusers.models import AutoencoderKL
from train_utils import parse_transport_args
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
import sys
from glob import glob
from evaluation.inception import InceptionV3
from models_assemble import FlowTurboAssemble


def create_npz_from_sample_folder(sample_dir, num=50_000, batch_size=4):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    activations = []
        
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        feature = np.load(f"{sample_dir}/{i:06d}.npy")
        activations.append(feature)

    activations = np.concatenate(activations)
    assert activations.shape == (num, 2048)
    npz_path = f"{sample_dir}.npz" # save both samples and statistics
    mu = np.nanmean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    np.savez(npz_path, activations=activations, mu=mu, sigma=sigma)
    print(f"Saved .npz file to {npz_path} [shape={activations.shape}].")
    return npz_path

@torch.no_grad()
def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    # random seed [current timestamp]
    # import time
    # current_time = int(time.time())
    # torch.manual_seed(current_time)

    torch.manual_seed(seed)

    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Load model:
    latent_size = args.image_size // 8
    # Create folder to save samples:
    tag = args.tag
    vae_ckpt = args.vae_ckpt
    METHED = {
        "N_H": 1,
        "N_P": 5,
        "N_R": 3,
    }
    sitturbo = FlowTurboAssemble(predictor_ckpt=args.predictor_ckpt, refiner_ckpt=args.refiner_ckpt, vae_ckpt=vae_ckpt, **METHED)
    sitturbo.eval() # important!

    image_size = "256"
    latent_size = int(image_size) // 8
    vae = AutoencoderKL.from_pretrained(vae_ckpt).to(device)
    sitturbo.to(device) # important!
    inception = InceptionV3().to(device).eval()
    
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0
    if not using_cfg:
        print('not use cfg')

    method_str = ''.join([f"{key}{value}" for key, value in METHED.items()])
    folder_name = f"{tag}-cfg-{args.cfg_scale}-{args.per_proc_batch_size}-fid{args.num_fid_samples}-{method_str}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    num_samples = len([name for name in os.listdir(sample_folder_dir) if (os.path.isfile(os.path.join(sample_folder_dir, name)) and ".png" in name)])
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    done_iterations = int( int(num_samples // dist.get_world_size()) // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0

    
    for i in pbar:
        # Sample inputs:
        z = torch.randn(n, 4, latent_size, latent_size, device=device)
        y = torch.randint(0, args.num_classes, (n,), device=device)
        
        # Setup classifier-free guidance:
        if using_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * n, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
        else:
            model_kwargs = dict(y=y)


        with torch.autocast(device_type="cuda"):
            imgs, samples = sitturbo(z, **model_kwargs)
        
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

        samples = vae.decode(samples / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255)
        # now [0, 255]
        inception_feature = inception(samples / 255.).cpu().numpy()
        
        # Save samples to disk as individual .png files
        index = rank + total
        np.save(f"{sample_folder_dir}/{index:06d}.npy", inception_feature)
        total += global_batch_size
        dist.barrier()

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        # Make sure all processes have finished saving their samples before attempting to convert to .npz
        def get_all_filenames_in_folder(folder_path):
            if not os.path.isdir(folder_path):
                print(f"Error: {folder_path} is an illegal path. ")
                return []
            filenames = os.listdir(folder_path, )
            return filenames
        sample_dir = sample_folder_dir + '/'
        filenames = get_all_filenames_in_folder(sample_dir)

        def create_npz_from_sample_folder(sample_dir, num=args.num_fid_samples, batch_size=50*4):
            """
            Builds a single .npz file from a folder of .png samples.
            """
            activations = []
            cnt = 0
            for name in tqdm(filenames):
                feature = np.load(sample_dir+name)
                activations.append(feature)
                cnt += 1

            activations = np.concatenate(activations)
            print(activations.shape)
            assert activations.shape == (num, 2048)
            npz_path = f"{folder_name}.npz" # save both samples and statistics
            mu = np.mean(activations, axis=0)
            sigma = np.cov(activations, rowvar=False)
            np.savez(npz_path, activations=activations, mu=mu, sigma=sigma)
            print(f"Saved .npz file to {npz_path} [shape={activations.shape}].")
            return npz_path
        print(filenames)
        create_npz_from_sample_folder(sample_dir)
        print("Done.")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--tag", type=str)
    parser.add_argument("--per_proc_batch_size", type=int, default=5)
    parser.add_argument("--num_fid_samples", type=int, default=5_0000)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg_scale",  type=float, default=4.0)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    
    parser.add_argument("--predictor_ckpt", type=str, default=None)     
    parser.add_argument("--refiner_ckpt", type=str, default=None)     
    parser.add_argument("--vae_ckpt", type=str, default=None)     
    
    parse_transport_args(parser)

    args = parser.parse_known_args()[0]
    main(args)
