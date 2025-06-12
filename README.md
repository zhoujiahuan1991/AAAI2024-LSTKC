# [AAAI2024] LSTKC: Long Short-Term Knowledge Consolidation for Lifelong Person Re-Identification 
<p align="center">
<a href="https://github.com/zhoujiahuan1991/AAAI2024-LSTKC"><img src="https://hitscounter.dev/api/hit?url=https%3A%2F%2Fgithub.com%2Fzhoujiahuan1991%2FAAAI2024-LSTKC&label=LSTKC&icon=github&color=%233d8bfd"></a>
</p>

The *official* repository for [LSTKC: Long Short-Term Knowledge Consolidation for Lifelong Person Re-Identification](https://ojs.aaai.org/index.php/AAAI/article/view/29554).

## News
* 🔥[2024.02.05] The code for LSTKC (accepted by AAAI 2024) is released!
* 🔥[2024.03.24] The full paper for LSTKC is publicly available!
* 🔥[2025.05.19] Our improved verison LSTKC++ is accepted by IEEE TPAMI. The full paper is available in [LSTKC++ Paper](https://ieeexplore.ieee.org/abstract/document/11010188/)!
* 🔥[2025.06.12] The code for LSTKC++ is released in [LSTKC++ Code](https://github.com/zhoujiahuan1991/LSTKC-Plus-Plus).
  
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
python continual_train.py --data-dir path/to/PRID
(for example, `python continual_train.py --data-dir ../DATA/PRID)
```

Evaluation from checkpoint:
```shell
python continual_train.py --data-dir path/to/PRID --test_folder /path/to/pretrained/folder --evaluate
```

## Results
The following results were obtained with a single NVIDIA 4090 GPU:

![Results](figs/result.png)

## Citation
If you find this code useful for your research, please cite our paper.
```shell
@inproceedings{xu2024lstkc,
  title={LSTKC: Long Short-Term Knowledge Consolidation for Lifelong Person Re-identification},
  author={Xu, Kunlun and Zou, Xu and Zhou, Jiahuan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={14},
  pages={16202--16210},
  year={2024}
}
```


## Acknowledgement
Our code is based on the PyTorch implementation of [PatchKD](https://github.com/feifeiobama/PatchKD) and [PTKP](https://github.com/g3956/PTKP).

## Contact

For any questions, feel free to contact us (xkl@stu.pku.edu.cn).

Welcome to our Laboratory Homepage ([OV<sup>3</sup> Lab](https://zhoujiahuan1991.github.io/)) for more information about our papers, source codes, and datasets.

