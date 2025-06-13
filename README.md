<div align="center">

# *SegFace* : Face Segmentation of Long-Tail Classes
<h3><strong>[Official Github](https://github.com/Kartik-3004/SegFace)</strong></h3>

</div>
<hr />

# Framework
<p align="center" width="100%">
  <img src='docs/static/images/segface.png' height="75%" width="75%">
</p>
Figure 2. 전체적인 구조. 이미지를 입력으로 넣어 Backbone network을 통해 multi-scale feture를 추출하고 MLP와 최종 decoder를 거쳐 class 별 segmentation 수행. 해당 아키텍처에서의 Backbone network을 바탕으로 아래의 pretrained model download 및 training 진행.

# Installation
```bash
git clone https://github.com/Kartik-3004/SegFace
cd SegFace

conda env create --file environment.yml
conda activate segface

#.env file 을 현재 main 디렉토리 (SegFace) 생성 및 LOG_PATH, DATA_PATH, ROOT_PATH를 .env file에 setup
# 아래 예시.
# DATA_PATH: Path to your dataset folder.
# ROOT_PATH: Path to your code directory.
# LOG_PATH: Path where the model checkpoints are stored and the training is logged.

# ex )
touch .env
echo 'ROOT_PATH=../SegFace' >> .env
echo 'DATA_PATH=../SegFace/data' >> .env
echo 'LOG_PATH=../SegFace/ckpts' >> .env
```

# Data (예시)
Open Data CelebAMask-HQ:<br>
[CelebAMask-HQ](https://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html)<br>
위 모델은 해당 데이터를 기반으로 학습한 것으로 custom data도 해당 방식으로 구성

아래의 구조로 데이터셋 디렉토리 구성:
```python
[DATA_PATH]/SegFace/
├── CelebAMask-HQ/
│   ├── CelebA-HQ-img/
│   ├── CelebAMask-HQ-mask-anno/
    └── list_eval_partition.txt

```
- CelebA-HQ-img : 실제 이미지 폴더
- CelebAMask-HQ-mask-anno: 각 이미지 별 얼굴 부위에 대한 mask 이미지
- list_eval_partition.txt: train/val/test 분할 정보 ex) 00001 0   # 각 이미지 ID에 대해 0=train, 1=val, 2=test

가령 아래의 이미지에 대해 각 부위에 대한 마스크 별로 필요함. 

(input image)
<p width="100%">
  <img src='https://github.com/user-attachments/assets/97855ae0-9285-4e00-9563-bb335479dc2f' height="30%" width="30%">
</p>

(input image에 대한 mask-anno)

![image](https://github.com/user-attachments/assets/a81ab117-bcaa-4314-80e2-1520163afd97)

(ex. hair)
<p width="100%">
  <img src='https://github.com/user-attachments/assets/0a31b790-b4c2-4b5a-b912-5340709df241' height="50%" width="50%">
</p>

(ex. nose)
<p width="100%">
  <img src='https://github.com/user-attachments/assets/6fd45cb4-fb6e-4277-bcfe-bb92f7eacaf2' height="50%" width="50%">
</p>


Pretrained weight from huggingface
| Arch | Resolution | Dataset         | Link                                                                            | Mean F1 |
|------|------------|-----------------|---------------------------------------------------------------------------------|---------|
| ConvNext  | 512 | CelebAMask-HQ     | [HuggingFace](https://huggingface.co/kartiknarayan/SegFace/tree/main/convnext_celeba_512) | 89.22 |



# Download Model weights
The pre-traind model can be downloaded manually from [HuggingFace](https://huggingface.co/kartiknarayan/SegFace) or using python:
```python
from huggingface_hub import hf_hub_download

# The filename "convnext_celeba_512" indicates that the model has a convnext bakcbone and trained
# on celeba dataset at 512 resolution.
hf_hub_download(repo_id="kartiknarayan/SegFace", filename="convnext_celeba_512/model_299.pt", local_dir="./weights")
hf_hub_download(repo_id="kartiknarayan/SegFace", filename="efficientnet_celeba_512/model_299.pt", local_dir="./weights")
hf_hub_download(repo_id="kartiknarayan/SegFace", filename="mobilenet_celeba_512/model_299.pt", local_dir="./weights")
hf_hub_download(repo_id="kartiknarayan/SegFace", filename="resnet_celeba_512/model_299.pt", local_dir="./weights")
hf_hub_download(repo_id="kartiknarayan/SegFace", filename="swinb_celeba_224/model_299.pt", local_dir="./weights")
hf_hub_download(repo_id="kartiknarayan/SegFace", filename="swinb_celeba_256/model_299.pt", local_dir="./weights")
hf_hub_download(repo_id="kartiknarayan/SegFace", filename="swinb_celeba_448/model_299.pt", local_dir="./weights")
hf_hub_download(repo_id="kartiknarayan/SegFace", filename="swinb_celeba_512/model_299.pt", local_dir="./weights")
hf_hub_download(repo_id="kartiknarayan/SegFace", filename="swinb_lapa_224/model_299.pt", local_dir="./weights")
hf_hub_download(repo_id="kartiknarayan/SegFace", filename="swinb_lapa_256/model_299.pt", local_dir="./weights")
hf_hub_download(repo_id="kartiknarayan/SegFace", filename="swinb_lapa_448/model_299.pt", local_dir="./weights")
hf_hub_download(repo_id="kartiknarayan/SegFace", filename="swinb_lapa_512/model_299.pt", local_dir="./weights")
hf_hub_download(repo_id="kartiknarayan/SegFace", filename="swinv2b_celeba_512/model_299.pt", local_dir="./weights")
```

# Usage
Download the trained weights from [HuggingFace](https://huggingface.co/kartiknarayan/SegFace) and ensure the data is downloaded with appropriate directory structure.<br>

### Training
```python
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=29440 /data/knaraya4/SegFace/train.py \
    --ckpt_path ckpts \
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
The trained models are stored at [LOG_PATH]/<ckpt_path>/<expt_name>.<br>
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
    --model_path [LOG_PATH]/<ckpt_path>/<expt_name>/model_299.pt


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
@inproceedings{narayan2025segface,
  title={Segface: Face segmentation of long-tail classes},
  author={Narayan, Kartik and Vs, Vibashan and Patel, Vishal M},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={6},
  pages={6182--6190},
  year={2025}
}
```

## Contact
If you have any questions, please create an issue on this repository or contact at knaraya4@jhu.edu
