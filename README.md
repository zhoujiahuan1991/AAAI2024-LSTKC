# [AAAI2024] LSTKC: Long Short-Term Knowledge Consolidation for Lifelong Person Re-Identification 
The *official* repository for  [LSTKC: Long Short-Term Knowledge Consolidation for Lifelong Person Re-Identification](xxx).

![Framework](figs/framework.png)

## Installation
```shell
conda create -n IRL python=3.7
conda activate IRL
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirement.txt
```
## Prepare Datasets
Download the person re-identification datasets [Market-1501](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view), [MSMT17](http://www.pkuvmc.com/dataset.html), [CUHK03](https://github.com/zhunzhong07/person-re-ranking/tree/master/evaluation/data/CUHK03), [SenseReID](https://drive.google.com/file/d/0B56OfSrVI8hubVJLTzkwV2VaOWM/view?resourcekey=0-PKtdd5m_Jatmi2n9Kb_gFQ). Other datasets can be prepared following [Torchreid_Datasets_Doc](https://kaiyangzhou.github.io/deep-person-reid/datasets.html) and [light-reid](https://github.com/wangguanan/light-reid).
Then unzip them and rename them under the directory like
```
PRID
├── CUHK01
│   └──..
├── CUHK02
│   └──..
├── CUHK03
│   └──..
├── CUHK-SYSU
│   └──..
├── DukeMTMC-reID
│   └──..
├── grid
│   └──..
├── i-LIDS_Pedestrain
│   └──..
├── MSMT17_V2
│   └──..
├── Market-1501
│   └──..
├── prid2011
│   └──..
├── SenseReID
│   └──..
└── viper
    └──..
```



## Quick Start
Training + evaluation:
```shell
`python continual_train.py --data-dir path/to/PRID`
(for example, `python continual_train.py --data-dir ../DATA/PRID`)
```

Evaluation from checkpoint:
```shell
`python continual_train.py --data-dir path/to/PRID --test_folder /path/to/pretrained/folder --evaluate`
```

## Results
The following results were obtained with a single NVIDIA 4090 GPU:

![Results](figs/result.png)

## Citation
If you find this code useful for your research, please cite our paper.

[1] Kunlun Xu, Xu Zou, Jiahuan Zhou. LSTKC: Long Short-Term Knowledge Consolidation for Lifelong Person Re-Identification[C]//Proceedings of the AAAI conference on artificial intelligence. 2024. 

## Acknowledgement
Our code is based on the PyTorch implementation of [PatchKD](https://github.com/feifeiobama/PatchKD) and [PTKP](https://github.com/g3956/PTKP).

## Contact

For any questions, feel free to contact us (xkl@stu.pku.edu.cn).

Welcome to our [Laboratory Homepage](http://www.icst.pku.edu.cn/mipl/home/) and [OV<sup>3</sup> Lab](https://zhoujiahuan1991.github.io/) for more information about our papers, source codes, and datasets.

