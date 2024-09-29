## 🚀 FlowTurbo: Towards Real-time Flow-Based Image Generation with Velocity Refiner (NeurIPS 2024)

Created by [Wenliang Zhao](https://wl-zhao.github.io/)\*, [Minglei Shi](https://github.com/shiml20)\*, [Xumin Yu](https://yuxumin.github.io/), [Jie Zhou](https://scholar.google.com/citations?user=6a79aPwAAAAJ&hl=en&authuser=1), [Jiwen Lu](https://scholar.google.com/citations?user=TN8uDQoAAAAJ&hl=en&authuser=1)

This repo contains PyTorch model definition and training/sampling code for FlowTurbo. 

[[arXiv]](http://arxiv.org/abs/2409.18128)

We propose a framework called FlowTurbo to accelerate the sampling of flow-based models while still enhancing the sampling quality. Our primary observation is that the velocity predictor’s outputs in the flow-based models will become stable during the sampling, enabling the estimation of velocity via a lightweight velocity refiner. Additionally, we introduce several techniques, including a pseudo corrector and sample-aware compilation, to further reduce inference time. Since FlowTurbo does not change the multi-step sampling paradigm, it can be effectively applied to various tasks such as image editing, inpainting, etc. By applying FlowTurbo to different flow-based models, we obtain an acceleration ratio of 53.1%∼58.3% on class-conditional generation and 29.8%∼38.5% on text-to-image generation. Notably, FlowTurbo reaches an FID of 2.11 on ImageNet with 100 (ms / img) and FID of 3.93 with 38 (ms / img), achieving the real-time image generation and establishing the new state-of-the-art.



## Setup

First, download and set up the repo:

```bash
git clone https://github.com/shiml20/FlowTurbo.git
cd FlowTurbo
```



We provide an [`environment.yml`](environment.yml) file that can be used to create a Conda environment. If you only want 
to run pre-trained models locally on CPU, you can remove the `cudatoolkit` and `pytorch-cuda` requirements from the file.

```bash
conda env create -f environment.yml
conda activate FlowTurbo
```



## Sampling 

**Pre-trained FlowTurbo checkpoints.** You can sample from our pre-trained FlowTurbo models with [`sample.py`](sample.py). Please Download the weights for our pre-trained FlowTurbo model.

```bash
python sample.py
```

For convenience, our pre-trained SiT-XL/2-Refiner models can be downloaded directly here as well:

| SiT-Refiner Model                                            | Image Resolution |
| ------------------------------------------------------------ | ---------------- |
| [XL/2](https://cloud.tsinghua.edu.cn/f/3d07d92dd2314857ae50/?dl=1) | 256x256          |



## Training SiT

We provide a training script for FlowTurbo in [`train.py`](train.py). To launch FlowTurbo (256x256) training with `N` GPUs on 
one node:

```bash
CUDA_VISIBLE_DEVICES='0' torchrun --nnodes=1 --nproc_per_node=1 --master_port 12345 train.py \
    --data-path /data/ILSVRC2012/train --global-batch-size 18 \
    --note 'NAME' --ckpt-every 5000 --lr 5e-5 --vae_ckpt vae-ema --model_teacher_ckpt /pretrained_models/predictor.ckpt \
```



## Evaluation (FID, Inception Score, etc.)

We include a `sample_ddp_feature.py` script which samples a large number of images from a SiT model in parallel. This script 
generates a folder of samples as well as a `.npz` file which can be directly used in `evaluator.py` which we provided to compute FID, Inception Score and other metrics. For example, to sample 50K images from our pre-trained FlowTurbo model over `N` GPUs under default ODE sampler settings, run:

```bash
torchrun --nnodes=1 --nproc_per_node=N sample_ddp_feature.py \
    --vae_ckpt vae-ema \
    --predictor_ckpt SiT-XL-2-256x256.pt \
    --refiner_ckpt SiT-XL-2-Refiner.pt \
    --num_fid_samples 60 --per_proc_batch_size 20 --cfg_scale 1.5 \
    --tag TEST ;\
```



##  Acknowledgments

We would like to express our sincere thanks to the author of [SiT](https://github.com/willisma/SiT) for the clear code base.

## Citation

If you find our work useful in your research, please consider citing:
```
@article{zhao2024flowturbo,
  title={FlowTurbo: Towards Real-time Flow-Based Image Generation with Velocity Refiner},
  author={Zhao, Wenliang and Shi, Minglei and Yu, Xumin and Zhou, Jie and Lu, Jiwen},
  journal={NeurIPS},
  year={2024}
}
```




## License

This project is under the MIT license. See [LICENSE](LICENSE.txt) for details.
