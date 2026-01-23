# PaGKD
<div align="center">
<h1>PaGKD</h1>
<h3>Pairing-free Group-level Knowledge Distillation for Robust Gastrointestinal Lesion Classification in White-Light Endoscopy</h3>
<br>
<a href="https://scholar.google.com/citations?user=rU2JxLIAAAAJ&hl=en">Qiang Hu</a><sup><span>1, &#42</span></sup>, Qimei Wang</a><sup><span>1, &#42</span></sup>, Yingjie Guo</a><sup><span>1</span></sup>, <a href="http://faculty.hust.edu.cn/liqiang15/zh_CN/index.htm">Qiang Li</a><sup><span>1, &#8224;</span></sup>, <a href="https://scholar.google.com/citations?user=LwQcmgYAAAAJ&hl=en">Zhiwei Wang</a><sup><span>1, &#8224;</span></sup>
</br>

<sup>1</sup>  WNLO, HUST
<br>
(<span>&#42;</span>: equal contribution, <span>&#8224;</span>: corresponding author)
</div>

## 1. Overview

Our framework PaGKD consists of a trainable WLI classifier and a frozen, pretrained NBI classifier. Both classifiers share the same architecture, following ADD (Hu et al. 2025). To facilitate knowledge transfer between unpaired WLI and NBI images, we introduce
a group-level feature distillation strategy, instead of regular image-level approaches. 

We then apply two complementary modules for multigranularity distillation: Group-level Prototype Knowledge Distillation (GKD-Pro) performs alignment of global classlevel distributions, while Group-level Dense Knowledge Distillation (GKD-Den) aligns local feature.
<p align="center">
<img src="https://github.com/kyo22/PaGKD/blob/main/img/method.jpg" alt="Image" width="1000px">
<p>

## 2. Checkpoints
| Model | PICCOLO (AUC) | IH-GC (AUC) | Weights (5-folds) |
| :---- | :------: | :------: | :------: |
| Ours | 0.901 | 0.840 | [ckpts](https://drive.google.com/drive/folders/1ohYcGtadxcfevFdtLmDsfntqxkdrQR2c?usp=drive_link) |
| w/o GKD-Pro & GKD-Den | 0.712 | 0.669 | [ckpts](https://drive.google.com/drive/folders/1bYGm3jm2LBIWna16y9GCOl_2HELoKYHD?usp=drive_link) |
| w/o GKD-Den | 0.835 | 0.755 | [ckpts](https://drive.google.com/drive/folders/1DGt-KsndfDP_fsgIzK5ZkUjc1JS-R16K?usp=drive_link) |
| w/o GKD-Pro | 0.850 | 0.773 | [ckpts](https://drive.google.com/drive/folders/12kqBsMzmgjs_ctETKoVPEsxbHTsyR6bw?usp=drive_link) |
<!---
| w/o LR-QFormer | 0.857 | 0.783 | [ckpts](https://drive.google.com/drive/folders/1cJ-kmwF0j0hSHgWcOUwz0YVxz7Ywia58?usp=drive_link) |
| w/o SRCA | 0.832 | 0.768 | [ckpts](https://drive.google.com/drive/folders/1OofX_Txw-_nEUVVhFNNmltfNo1REEw_L?usp=drive_link) |
| w/o Bidirectional | 0.878 | 0.819 | [ckpts](https://drive.google.com/drive/folders/1OofX_Txw-_nEUVVhFNNmltfNo1REEw_L?usp=drive_link) |
-->
## 3. Visulization of Results
### 3.1 ROC Curve:
<p align="center">
<img src="https://github.com/kyo22/PaGKD/blob/main/img/roc.png" alt="Image" width="800px">
<p>

### 3.2 t-SNE:
Comparisons between our proposed group-levelm distillation components and their image-level variants.
<p align="center">
<img src="https://github.com/Huster-Hq/ADD/blob/main/imgs/CAM_visualization.png" alt="Image" width="600px">
<p>


## 4. Getting Started
### 4.1 Recommended Environment:
- Python 3.8+
- PyTorch 2.1+ 
- TorchVision corresponding to the PyTorch version
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Install other dependent packages:
```
cd ADD
pip install -r requirements.txt
```

### 4.2 Data Preparation
- Downloading the [CPC-Paired dataset](https://github.com/WeijieMax/CPC-Trans) (public WLI-NBI paired polyp classification dataset). The file paths should be arranged as follows:
```
ADD
├── dataset
├── ├── White_light
├── ├── ├── adenomas
├── ├── ├── ├── ├── 01-1.png
├── ├── ├── ├── ├── 02-1.png
├── ├── ├── ├── ├── ......
├── ├── ├── hyperplastic_lesions
├── ├── ├── ├── ├── 011-1.png
├── ├── ├── ├── ├── 011-2.png
├── ├── ├── ├── ├── ......
├── ├── NBI
├── ├── ├── adenomas
├── ├── ├── ├── ├── 01-1.png
├── ├── ├── ├── ├── 02-1.png
├── ├── ├── ├── ├── ......
├── ├── ├── hyperplastic_lesions
├── ├── ├── ├── ├── 011-1.png
├── ├── ├── ├── ├── 011-2.png
├── ├── ├── ├── ├── ......
```

- Note that the details of dataset splitation in the 5-fold experiment can be downloaded in [here](https://drive.google.com/drive/folders/1UkLZxZDGyKH3P3TIAra-tORzBEuwx-3E?usp=drive_link). You need to download these `.txt` files and put them into a newly created folder `split` and the file paths should be arranged as follows:
```
ADD
├── split
├── ├── xxx.txt
├── ├── ......
```


### 4.3 Training:
Stage 1: pre-traning the NBI classifier:
```
python train_teacher.py
```
Stage 2: training the WLI classifier:
```
python train.py
```

### 4.4 Testing and Evaluation:
```
python test.py
```
You can also directly download the `well-trained model` from [Google Drive](https://drive.google.com/drive/folders/1ohYcGtadxcfevFdtLmDsfntqxkdrQR2c?usp=drive_link), and predict the results by `test.py`.

## Citation
If you find our paper and code useful in your research, please consider giving us a star ⭐ and citing PaGKD by the following BibTeX entry.
```
@article{hu2026pairing,
  title={Pairing-free Group-level Knowledge Distillation for Robust Gastrointestinal Lesion Classification in White-Light Endoscopy},
  author={Hu, Qiang and Wang, Qimei and Guo, Yingjie and Li, Qiang and Wang, Zhiwei},
  journal={arXiv preprint arXiv:2601.09209},
  year={2026}
}
```
