> # Patche_based_synthesis

문제 정의.
특정 이미지를 M x N 크기로 분할한 후에 부분 이미지를 임의로 변환(mirror, flip, rotate)하고, 해당 부분 이미지들을 가지고 원 이미지를 구성하는 것.

```
Algorithm.
1. 이미지의 shape을 동일하게 만드는 과정(rotation 된 patches를 다른 shape과 동일하게 90도 rotation 진행)
2. 가정1 : 가로와 세로 크기가 주어짐.
3. 세로 edge들을 이어 붙이는 작업을 (세로 크기 - 1)만큼 반복.
4. 특정 measure를 가지고 가능한 모든 경우의 수를 탐색하며 진행.
5. M개의 개수만큼의 이미지가 생성되었기에 가로 edge들을 이어 붙이는 작업을 (가로 크기 - 1)만큼 반복.
```

> # Measure

이미지를 연결하고 연결된 이미지가 자연스러운지 파악하는 방법으로 픽셀 값 차이, 구조적 유사도(SSIM), 히스토그램 기반 유사도, 구조 텐서 유사도(STSIM) 등을 사용 하였음.

> # 한계 및 오류

- 해결 가능한 문제
  1. image_patches들이 정사각형으로 나누어 지는 경우 확인해야 하는 경우의 수가 늘어남.
     - root가 되는 이미지 자체도 돌아가 있을 수 있기 때문에 해당경우도 신경을 써야함.
     - (상하좌우 모두 비교하는 코드를 짜면 해결이 됨)
  2. 
  3. make_patches를 진행하여 저장한 이미지를 불러오는 데이터와 원본 이미지를 자른 데이터 간의 미세한 차이가 존재.
      - (cv2 안에 포함된 함수를 자세히 확인해 보면 파악 가능)

- 불가능한 문제
  - measure : SSIM 으로도 일정 이상의 성능을 내기는 어려움.(다른 대안을 모색해야 함)
    - Idea1 : 이미지는 일종의 sequence 데이터로 볼 수 있어 해당 내용을 확률적 모델(MCMC 등)을 추출 방법이나 신경망 기반의 학습을 통해 예측하는 방법(?)

---

```
├── image/
├── image_patches
├── merge_patches
├── cut_image.py/make_patches(class)
│   ├── __init__(def) : class의 전체적인 작업 진행
│   ├── seed(def) : test를 위한 random seed 설정
│   ├── split_image(def) : 이미지 M x N 크기로 분할
│   ├── random_mirror(def) : 임의의 mirror 변환
│   ├── random_flip(def) : 임의의 flip 변환
│   ├── random_rotation(def) : 임의의 rotation 변환
│   └── process_images(def) : 분할된 이미지에 임의의 변환 작업들 진행.
├── merge_image.py/merge_patches(class)
│   ├── __init__(def) : class의 전체적인 작업 진행
│   ├── seed(def) : test를 위한 random seed 설정
│   ├── rotate_image(def) : 이미지 rotation 변환
│   ├── process_image(def) : 이미지 shape를 맞추기위한 작업 
│   ├── mirror(def) : 이미지 mirror 변환
│   ├── flip(def) : 이미지 flip 변환
│   ├── calculate sim(def) : 이미지간 유사도 측정
│   ├── col_combine(def) : 이미지 세로 edge 결합
│   └── row_combine(def) : 이미지 가로 edge 결합
├── option.py
│   └── parse_opt(def) : option 기본 설정.
├── Pipfile
├── Pipfile.lock
├── temp.ipynb
│   ├── test image1 : image 1을 가지고 test한 결과
│   ├── test image2 : image 2를 가지고 test한 결과
│   └── error
└── README.md
```

