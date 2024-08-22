# RDesign: Hierarchical Data-efficient Representation Learning for Tertiary Structure-based RNA Design

![GitHub stars](https://img.shields.io/github/stars/A4Bio/RDesign)  ![GitHub forks](https://img.shields.io/github/forks/A4Bio/RDesign?color=green) 
<a href="https://colab.research.google.com/drive/1mE3kYaXFvAnsc_zULYm5InO_kxlQBKZ3#scrollTo=3aJDuVNJFic2" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

**[2024-08-21] News:** We provided a comprehensive evaluation system for RNA sequence design and prediction named **R3Design**. APIs and Colab demos are also provided. Feel free to check out our new [repo](https://github.com/A4Bio/R3Design)!

**[2024-08-15] Update:** Thank you all for the interests and inquries about our paper, we are sorry that we haven't provided detailed documentation and demo of the paper for such a long time. Now, it has been solved. Feel free to check out our updated documentation and colab! :)   
## Introduction

While artificial intelligence has made remarkable strides in revealing the relationship between biological macromolecules' primary sequence and tertiary structure, designing RNA sequences based on specified tertiary structures remains challenging. Though existing approaches in protein design have thoroughly explored structure-to-sequence dependencies in proteins, RNA design still confronts difficulties due to structural complexity and data scarcity.

In this study, we aim to systematically construct a data-driven RNA design pipeline. We crafted a large, well-curated benchmark dataset and designed a comprehensive structural modeling approach to represent the complex RNA tertiary structure. More importantly, we proposed a hierarchical data-efficient representation learning framework that learns structural representations through contrastive learning at both cluster-level and sample-level to fully leverage the limited data. Extensive experiments demonstrate the effectiveness of our proposed method, providing a reliable baseline for future RNA design tasks.

<p align="center">
  <img src='https://github.com/A4Bio/RDesign/assets/34480960/ef6029bb-95e1-4ea9-a965-fefb68b2e102' width="100%">
</p>


## Dataset

We carefully collected representative RNA tertiary structure data from two sources, RNAsolo and the Protein Data Bank (PDB). The refined data has been released [here](https://github.com/A4Bio/RDesign/releases/tag/data). Please download the datasets and organize them as follows.

```
RDesign
├── API
├── assets
├── checkpoints
├── methods
├── model
└── data
    ├── RNAsolo
    │   ├── train_data.pt
    │   ├── val_data.pt
    │   ├── test_data.pt
```

### Main Environment

```shell
cd RDesign
conda env create -f environment.yml
conda activate RDesign
```

### Load Data

```shell
# If you want to see the details inside our dataset, you could use Pickle package from Python
import _pickle as cPickle
train_data = cPickle.load(open('data/train_data.pt', 'rb'))
print(train_data[0].keys())

#For external datasets, loading data could be in this way:
from API.rpuzzles_dataset import RPuzzlesDataset
rfam_dataset = RPuzzlesDataset('./data/rfam_data.pt')
rpuz_dataset = RPuzzlesDataset('./data/rpuz_data.pt')
```

### Test the model

```shell
# For more details, please refer to the colab
# We provided detailed functions and pipeline to show how our model operates
```
Colab Link: 

<a href="https://colab.research.google.com/drive/1mE3kYaXFvAnsc_zULYm5InO_kxlQBKZ3#scrollTo=3aJDuVNJFic2" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


## Citation

If you are interested in our repository and our paper, please cite the following paper:

```
@inproceedings{tan2024rdesign,
  title={RDesign: Hierarchical Data-efficient Representation Learning for Tertiary Structure-based RNA Design},
  author={Tan, Cheng and Zhang, Yijie and Gao, Zhangyang and Hu, Bozhen and Li, Siyuan and Liu, Zicheng and Li, Stan Z},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024}
}
```

## Feedback
If you have any issue about this work, please feel free to contact me by email: 
* Cheng Tan: tancheng@westlake.edu.cn
* Yijie Zhang: yj.zhang@mail.mcgill.ca

## License

This project is released under the [Apache 2.0 license](LICENSE). See `LICENSE` for more information.
