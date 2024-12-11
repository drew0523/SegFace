<div align="center">

# *SegFace* : Face Segmentation of Long-Tail Classes
<h3><strong>AAAI 2025</strong></h3>

[Kartik Narayan](https://kartik-3004.github.io/portfolio/) &emsp; [Vibashan VS](https://vibashan.github.io) &emsp; [Vishal M. Patel](https://engineering.jhu.edu/faculty/vishal-patel/)

Johns Hopkins University

<a href='https://kartik-3004.github.io/SegFace/'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
<a href='https://kartik-3004.github.io/SegFace/'><img src='https://img.shields.io/badge/Paper-arXiv-red'></a>
<a href='https://huggingface.co/kartiknarayan/SegFace'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-orange'></a>

</div>
<hr />

## Contributions

<p align="center" width="100%">
  <img src='docs/static/images/qualitative_results_1.png' height="75%" width="75%">
</p>

Figure 1. The qualitative comparison highlights the superior performance of our method, <i>SegFace</i>, compared to DML-CSR. In (a), SegFace effectively segments both long-tail classes like earrings and necklaces as well as head classes such as hair and neck. In (b), it also excels in challenging scenarios involving multiple faces, human-resembling features, poor lighting, and occlusion, where DML-CSR struggles.

The key contributions of our work are,<br>
1️⃣ We introduce a lightweight transformer decoder with learnable class-specific tokens, that ensures each token is dedicated to a specific class, thereby enabling independent modeling of classes. The design effectively addresses the challenge of poor segmentation performance of long-tail classes, prevalent in existing methods.<br>
2️⃣ Our multi-scale feature extraction and MLP fusion strategy, combined with a transformer decoder that leverages learnable class-specific tokens, mitigates the dominance of head classes during training and enhances the feature representation of long-tail classes.<br>
3️⃣ <i>SegFace</i> establishes a new state-of-the-art performance on the LaPa dataset (93.03 mean F1 score) and the CelebAMask-HQ dataset (88.96 mean F1 score). Moreover, our model can be adapted for fast inference by simply swapping the backbone with a MobileNetV3 backbone. The mobile version achieves a mean F1 score of 87.91 on the CelebAMask-HQ dataset with 95.96 FPS.<br>

> **<p align="justify"> Abstract:** *Face parsing refers to the semantic segmentation of human faces into
> key facial regions such as eyes, nose, hair, etc. It serves as a prerequisite for various advanced applications,
> including face editing, face swapping, and facial makeup, which often require segmentation masks for classes
> like eye-glasses, hats, earrings, and necklaces. These infrequently occurring classes are called long-tail
> classes, which are over-shadowed by more frequently occurring classes known as head classes. Existing methods,
> primarily CNN-based, tend to be dominated by head classes during training, resulting in suboptimal representation
> for long-tail classes. Previous works have largely overlooked the problem of poor segmentation performance of
> long-tail classes. To address this issue, we propose SegFace, a simple and efficient approach that uses a
> lightweight transformer-based model which utilizes learnable class-specific tokens. The transformer decoder
> leverages class-specific tokens, allowing each token to focus on its corresponding class, thereby enabling
> independent modeling of each class. The proposed approach improves the performance of long-tail classes, thereby
> boosting overall performance. To the best of our knowledge, SegFace is the first work to employ transformer models
> for face parsing. Moreover, our approach can be adapted for low-compute edge devices, achieving 95.96 FPS. We
> conduct extensive experiments demonstrating that SegFace significantly outperforms previous state-of-the-art models,
> achieving a mean F1 score of 88.96 (+2.82) on the CelebAMask-HQ dataset and 93.03 (+0.65) on the LaPa dataset.* </p>

# Framework
<p align="center" width="100%">
  <img src='docs/static/images/segface.png' height="75%" width="75%">
</p>
Figure 2. The proposed architecture, <i>SegFace</i>, addresses face segmentation by enhancing the performance on long-tail classes through a transformer-based approach. Specifically, multi-scale features are first extracted from an image encoder and then fused using an MLP fusion module to form face tokens. These tokens, along with class-specific tokens, undergo self-attention, face-to-token, and token-to-face cross-attention operations, refining both class and face tokens to enhance class-specific features. Finally, the upscaled face tokens and learned class tokens are combined to produce segmentation maps for each facial region.

# :rocket: News
- [12/09/2025] 🔥 We release *SegFace*.

# Installation
```bash
conda env create --file environment.yml
conda activate segface

# Create a .env file inside the main directory (SegFace) and setup LOG_PATH, DATA_PATH and ROOT_PATH in the .env file.
# Provided below is an example which we used as per our directory structure.
# DATA_PATH: Path to your dataset folder.
# ROOT_PATH: Path to your code directory.
# LOG_PATH: Path where the model checkpoints are stored and the training is logged.

touch .env
echo 'ROOT_PATH=/data/knaraya4/SegFace' >> .env
echo 'DATA_PATH=/data/knaraya4/data/SegFace' >> .env
echo 'LOG_PATH=/mnt/store/knaraya4/SegFace' >> .env
```

# Download Data
The datasets can be downloaded from their respective webpages or by mailing the authors:<br>
1. [CelebAMask-HQ](https://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html)<br>
2. [LaPa](https://github.com/jd-opensource/lapa-dataset)<br>
3. [Helen](https://github.com/zhfe99/helen)<br>

Arrange the dataset in the following manner:
```python
[DATA_PATH]/SegFace/
├── CelebAMask-HQ/
│   ├── CelebA-HQ-img/
│   ├── CelebA-HQ-to-CelebA-mapping.txt
│   ├── CelebAMask-HQ-attribute-anno.txt
│   ├── CelebAMask-HQ-mask-anno/
│   ├── CelebAMask-HQ-pose-anno.txt
│   ├── list_eval_partition.txt
│   └── README.txt
├── helen/
│   ├── f1_score.py
│   ├── label_names.txt
│   ├── landmarks.txt
│   ├── list_68pt_rect_attr_test.txt
│   ├── list_68pt_rect_attr_train.txt
│   ├── list_annos_trn.txt
│   ├── list_annos_tst.txt
│   ├── README.md
│   ├── test/
│   ├── test_resize/
│   └── train/
└── LaPa/
    ├── test/
    ├── train/
    └── val/
```

# Download Model weights
The pre-traind model can be downloaded manually from [HuggingFace](https://huggingface.co/kartiknarayan/segface) or using python:
```python
from huggingface_hub import hf_hub_download

# Finetuned Weights

# The filename "swin_arcface_webface4m_tinyface" indicates that the model has a swin bakcbone and pretraind
# on webface4m dataset with arcface loss function and finetuned on tinyface.
hf_hub_download(repo_id="kartiknarayan/petalface", filename="swin_arcface_webface4m_tinyface/model.pt", local_dir="./weights")
hf_hub_download(repo_id="kartiknarayan/petalface", filename="swin_cosface_webface4m_tinyface/model.pt", local_dir="./weights")
hf_hub_download(repo_id="kartiknarayan/petalface", filename="swin_cosface_webface4m_briar/model.pt", local_dir="./weights")
hf_hub_download(repo_id="kartiknarayan/petalface", filename="swin_cosface_webface12m_briar/model.pt", local_dir="./weights")

# Pre-trained Weights
hf_hub_download(repo_id="kartiknarayan/petalface", filename="swin_arcface_webface4m/model.pt", local_dir="./weights")
hf_hub_download(repo_id="kartiknarayan/petalface", filename="swin_cosface_webface4m/model.pt", local_dir="./weights")
hf_hub_download(repo_id="kartiknarayan/petalface", filename="swin_arcface_webface12m/model.pt", local_dir="./weights")
hf_hub_download(repo_id="kartiknarayan/petalface", filename="swin_cosface_webface12m/model.pt", local_dir="./weights")
```

# Usage
Download the trained weights from [HuggingFace](https://huggingface.co/kartiknarayan/segface) and ensure the data is downloaded with appropriate directory structure.<br>

### Training
```python
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=29440 /data/knaraya4/SegFace/train.py \
    --expt_name swin_base_celeba_512 \
    --dataset celebamask_hq \
    --backbone segface_celeb \
    --model swin_base \
    --lr 1e-4 \
    --lr_schedule 80,200 \
    --input_resolution 512 \
    --train_bs 4 \
    --val_bs 1 \
    --test_bs 1 \
    --num_workers 4 \
    --epochs 300

### You can change the model backbone by changing --model
# --model swin_base, swinv2_base, swinv2_small, swinv2_tiny
# --model convnext_base, convnext_small, convnext_tiny
# --model mobilenet
# --model efficientnet

### You can change the dataset on which the model is trained on by changing --dataset and --backbone
# CelebAMaskHQ: --model segface_celeb --dataset celebamask_hq
# LaPa: --model segface_lapa --dataset lapa
# Helen: --model segface_helen --dataset helen
```
The trained models are stored at [LOG_PATH]/<expt_name>.<br>
<b>NOTE</b>: The training scripts are provided at [SegFace/scripts](scripts).

### Inference
```python
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 python /data/knaraya4/SegFace/test.py \
    --ckpt_path ckpts \
    --expt_name <expt_name> \
    --dataset <dataset_name> \
    --backbone <backbone_name> \
    --model <model_name> \
    --input_resolution 512 \
    --test_bs 1 \
    --model_path /mnt/store/knaraya4/SegFace/ckpts/<expt_name>/model_299.pt


# --dataset celebamask_hq
# --dataset lapa
# --dataset helen

# --backbone segface_celeb
# --backbone segface_lapa
# --backbone segface_helen

# --model swin_base, swinv2_base, swinv2_small, swinv2_tiny
# --model convnext_base, convnext_small, convnext_tiny
# --model mobilenet
# --model efficientnet
```
<b>NOTE</b>: The inference script is provided at [SegFace/scripts](scripts).

## Citation
If you find *SegFace* useful for your research, please consider citing us:

```bibtex
Coming Soon !!!
```

## Contact
If you have any questions, please create an issue on this repository or contact at knaraya4@jhu.edu
