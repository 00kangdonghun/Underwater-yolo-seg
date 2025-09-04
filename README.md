# Underwater-segmentation

- 수중 환경에서의 객체 인식은 탁도, 빛의 흡수와 산란, 색 왜곡 등으로 인해 기존 YOLOv8-seg 모델의 성능이 저하됩니다.
- 본 프로젝트는 YOLOv8-seg 기반의 수중 특화 개선 모델을 제안하여, BBox와 Mask의 mAP 성능 향상을 목표로 합니다.


## 📚 Table of Contents
- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Setup](#-setup)
- [Usage](#-usage)
- [Results](#-results)
- [References](#-references)


## 🔍 Overview

본 프로젝트는 YOLOv8-seg의 **Backbone + Head** 구조를 수중 환경에 맞게 개선하여 객체 탐지 및 세그멘테이션 성능을 향상시킨 연구입니다.

개선된 주요 요소 : 
- **ImageEnhancementBlock(IEB)** -> 색 보정 + 대비 향상
- **ASPPBlock** -> multiscale 문맥 정보 활용
- **EfficientSEBlock** -> 채널 간 중요도
- **callback-uw** -> underwater image 전처리 및 증강 기법

Datasets used:
- [✔] 해양침적쓰레기 이미지 데이터 고도화 ("로프", "어망류" 데이터셋만 활용/ 31,314장)
https://www.aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=%ED%95%B4%EC%96%91%EC%B9%A8%EC%A0%81&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&aihubDataSe=data&dataSetSn=71340
- [✔] 연안어장 생태환경 피해 유발 해양생물 데이터구축 ("화질 개선후_polygon"데이터셋만 활용/ 185,352장)
https://www.aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=%EC%97%B0%EC%95%88%EC%96%B4%EC%9E%A5&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&aihubDataSe=data&dataSetSn=71328
- [✔] 하천 및 항만 수중생활 폐기물 영상데이터 ("거제"에 해당하는 데이터셋만 활용/ 115,270장)
https://www.aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=%ED%95%98%EC%B2%9C+%EB%B0%8F&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&aihubDataSe=data&dataSetSn=71293
- [✔] UIIS-10K (전체 데이터셋 활용, 10classes/ 10,048장)
https://github.com/LiamLian0727/UIIS10K/blob/main/README.md

*code에서 각 데이터셋은 아래와 같이 정의합니다.

해양침적쓰레기 이미지 데이터 고도화 = deposition_302

연안어장 생태환경 피해 유발 해양생물 데이터구축 = marine_262

하천 및 항만 수중생활 폐기물 영상데이터 = trash_295

## 📁 Project Structure

```
.
├── convert2yolo
│   ├── deposition302_convert2yolo.py  # deposition_302 convert2yolo
│   ├── marine262_convert2yolo.py     # marine_262 convert2yolo
│   └── trash295_convert2yolo.py      # trash_295 convert2yolo
│    
├── ultralytics/
│   ├── cfg/
│   │   ├── models/v8/uw-final.yaml     # 수중 특화 모델 정의
│   │   └── datasets/UIIS10K.yaml       # 데이터셋 설정 파일 (trash_295, deposition_302, marine_262)
│   ├── nn/modules/
│   │   ├── block.py                    # 신규 블록 추가 (IEB, ASPP, EfficientSE)
│   │   ├── conv.py
│   │   └── tasks.py                    # parse 함수 내 신규 블록 등록
│   └── utils/callbacks/callbacks_uw.py # 수중 학습 전용 콜백
│
├── train.py                             # train 실행 스크립트
├── test.py                              # test 실행 스크립트
├── requirements.txt
└── README.md
```


## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/underwater-segmentation.git
cd underwater-segmentation
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```


## 🚀 Usage

### 1. Json2YOLO
- deposition_302 / marine_262 / trash_295 dataset convert2yolo
```bash
cd convert2yolo
python deposition302_convert2yolo.py
python marine262_convert2yolo.py
python trash295_convert2yolo.py
```
- UIIS-10K dataset convert2yolo
```bash
from ultralytics.data.converter import convert_coco

convert_coco(labels_dir="/home1/danny472/Underwater/dataset/UIIS/UDW/annotations", use_segments=True)
```

### 2. Train
- Train
```bash
python train.py
```

### 3. Test
- Test
```bash
python test.py
```


## 📊 Results (example)

<img width="1029" height="728" alt="image" src="https://github.com/user-attachments/assets/ceb60cf9-0192-48a3-9b88-71c58e82542d" />


## 📖 References

- [WaterMask: Instance Segmentation for Underwater Imagery](https://openaccess.thecvf.com/content/ICCV2023/papers/Lian_WaterMask_Instance_Segmentation_for_Underwater_Imagery_ICCV_2023_paper.pdf)
- [Integer-Valued Training and Spike-Driven Inference Spiking Neural Network for High-performance and Energy-efficient Object Detection](https://arxiv.org/pdf/2407.20708)
...
