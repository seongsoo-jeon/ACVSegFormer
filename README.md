# Q-AVSegFormer
<img width="1009" alt="image" src="image/arch.png">
할것. 
1. 이름 정하기
2. 내용 설명 
3. architecture 사진 붙이기


## Updates
- (미래) document 완료하기
- (2025.12.17) We completed the implemention of Q-AVSegFormer.



## How To Start?

### 1. Requirements and Installation
## a. Want to Use Docker
```shell
# project clone
git clone https://github.com/seongsoo-jeon/avsegformer.git
cd avsegformer

# build image
docker build -t avsegformer . 

# run container (connect to container by cli && ./data to /app/AVSegFormer/data)
docker run --gpus all \
    -v $(pwd)/data:/app/AVSegFormer/data \
    -it avsegformer bash
```

## b. Local Environments
```shell
# project clone
git clone https://github.com/seongsoo-jeon/avsegformer.git
cd avsegformer

# recommended
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html
pip install pandas timm resampy soundfile

# build MSDeformAttention
cd ops
sh make.sh
```


### 2. Data Preparation
Please refer to the link [AVSBenchmark](https://github.com/OpenNLPLab/AVSBench) to download the datasets. You can put the data under `data` folder. Remember to modify the path in config files. The `data` directory is as bellow:
```
|--data
   |--Multi-sources
      |--ms3_meta
      |--ms3_meta_data.csv
   |--Single-source
      |--s4_data
      |--s4_meta_data.csv
   |--AVSS
```


### 3. Download Pre-Trained Models
- 이거 수정 필요

- The pretrained backbone is available from benchmark [AVSBench pretrained backbones](https://drive.google.com/drive/folders/1386rcFHJ1QEQQMF6bV1rXJTzy8v26RTV).
- We provides pre-trained models for all three subtasks. You can download them from [AVSegFormer pretrained models](https://drive.google.com/drive/folders/1ZYZOWAfoXcGPDsocswEN7ZYvcAn4H8kY).

|Method|Backbone|Subset|Lr schd|Config|mIoU|F-score|Download|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|AVSegFormer-R50|ResNet-50|S4|30ep|[config](config/s4/AVSegFormer_res50_s4.py)|76.38|86.7|[ckpt](https://drive.google.com/file/d/1nvIfR-1XZ_BgP8ZSUDuAsGhAwDJRxgC3/view?usp=drive_link)|
|AVSegFormer-PVTv2|PVTv2-B5|S4|30ep|[config](config/s4/AVSegFormer_pvt2_s4.py)|83.06|90.5|[ckpt](https://drive.google.com/file/d/1ZJ55jxoHP1ur-hLBkGcha8sjptE_shfw/view?usp=drive_link)|
|AVSegFormer-R50|ResNet-50|MS3|60ep|[config](config/ms3/AVSegFormer_res50_ms3.py)|53.81|65.6|[ckpt](https://drive.google.com/file/d/1MRk5gQnUtiWwYDpPfB20fO07SVLhfuIV/view?usp=drive_link)|
|AVSegFormer-PVTv2|PVTv2-B5|MS3|60ep|[config](config/ms3/AVSegFormer_pvt2_ms3.py)|61.33|73.0|[ckpt](https://drive.google.com/file/d/1iKTxWtehAgCkNVty-4H1zVyAOaNxipHv/view?usp=drive_link)|
|AVSegFormer-R50|ResNet-50|AVSS|30ep|[config](config/avss/AVSegFormer_res50_avss.py)|26.58|31.5|[ckpt](https://drive.google.com/file/d/1RvL6psDsINuUwd9V1ESgE2Kixh9MXIke/view?usp=drive_link)|
|AVSegFormer-PVTv2|PVTv2-B5|AVSS|30ep|[config](config/avss/AVSegFormer_pvt2_avss.py)|37.31|42.8|[ckpt](https://drive.google.com/file/d/1P8a2dJSUoW0EqFyxyP8B1-Rnscxnh0YY/view?usp=drive_link)|


### 4. Train
```shell
TASK = "s4"  # or ms3, avss
CONFIG = "config/s4/AVSegFormer_pvt2_s4.py"

bash train.sh ${TASK} ${CONFIG}
```


### 5. Test
```shell
TASK = "s4"  # or ms3, avss
CONFIG = "config/s4/AVSegFormer_pvt2_s4.py"
CHECKPOINT = "work_dir/AVSegFormer_pvt2_s4/S4_best.pth"

bash test.sh ${TASK} ${CONFIG} ${CHECKPOINT}
```


## Citation

If you use our model, please consider cite following papers:
```
@article{zhou2023avss,
      title={Audio-Visual Segmentation with Semantics}, 
      author={Zhou, Jinxing and Shen, Xuyang and Wang, Jianyuan and Zhang, Jiayi and Sun, Weixuan and Zhang, Jing and Birchfield, Stan and Guo, Dan and Kong, Lingpeng and Wang, Meng and Zhong, Yiran},
      journal={arXiv preprint arXiv:2301.13190},
      year={2023},
}

@misc{gao2023avsegformeraudiovisualsegmentationtransformer,
      title={AVSegFormer: Audio-Visual Segmentation with Transformer}, 
      author={Shengyi Gao and Zhe Chen and Guo Chen and Wenhai Wang and Tong Lu},
      year={2023},
      eprint={2307.01146},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
}

@~{우리꺼.. 논문??? ㅋㅋㅋ}
```
