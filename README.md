# DPMamba
DPMamba: Distillation Prompt Mamba for Multimodal Remote Sensing Image Classification with Missing Modalities

*Yueguang Yang, Jiahui Qu, Ling Huang, Wenqian Dong*


![Teaser Image](pic/framework.jpg)   

## Environment Setup

Please refer to the [VMamba installation instructions](https://github.com/MzeroMiko/VMamba) for the environment setup.


## Quick Start

### Data Preparation
1. The data can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1nbOzUDTT0GXN8VDpw7ldWTt7WG-NRYG_?usp=sharing).

2. Organize dataset structure:
```txt
datasets/
├── Houston/
│   ├── HSI.mat
│   ├── LiDAR.mat         
│   ├── All_label.mat
├── Trento/
│   ├── HSI.mat
│   ├── LiDAR.mat         
│   ├── All_label.mat
├── Augsburg/
│   ├── HSI.mat
│   ├── LiDAR.mat     
│   ├── SAR.mat       
│   ├── All_label.mat
└──
```
   

### Pre-training Stage
Download pre-trained text encoder from [Google Drive](https://drive.google.com/file/d/1lGomu2aL8PWBiPC-kq2NnCJjDbqAkYWX/view?usp=sharing)
and place it in the root directory.

```shell
python main_RS.py --cfg configs/<dataset name>.yaml  --is_Pretrain True 
```

### Training Stage
```shell
python main_RS.py --cfg configs/<dataset name>.yaml --is_Pretrain False --MODEL_PATH <pre-trained model path> 
```

[//]: # (## Citation)

[//]: # ()
[//]: # (If you find this project helpful, please use the following BibTeX entry.)

[//]: # (```BibTeX)

[//]: # (@article{lv2025test,)

[//]: # (  title={Test-Time Domain Generalization via Universe Learning: A Multi-Graph Matching Approach for Medical Image Segmentation},)

[//]: # (  author={Lv, Xingguo and Dong, Xingbo and Wang, Liwen and Yang, Jiewen and Zhao, Lei and Pu, Bin and Jin, Zhe and Li, Xuejun},)

[//]: # (  journal={arXiv preprint arXiv:2503.13012},)

[//]: # (  year={2025})

[//]: # (})

[//]: # (```)

## Acknowledgement

We gratefully acknowledge the following open-source projects that inspired or contributed to our implementation:

- [VMamba](https://github.com/MzeroMiko/VMamba)
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer)




