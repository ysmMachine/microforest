# MicroForest

`MicroForest`는 MDPI *Applied Sciences*에 게재된 논문
**MicroForest: Lightweight Bottleneck Prediction for Manufacturing Processes on Edge Devices**의 핵심 아이디어를 재구현한 연구용 코드입니다.

이 저장소는 논문 실험 전체를 그대로 복제하기보다는, 데이터 생성부터 모델 구성과 동작 확인까지 이어지는 최소 재현 파이프라인을 제공합니다.

> 논문 그림은 원문에서 추출했으며, 원문 라이선스(CC BY 4.0)에 따라 출처를 명시해 포함했습니다.  
> Source: Yoo, S.; Oh, C. *MicroForest: Lightweight Bottleneck Prediction for Manufacturing Processes on Edge Devices*. Applied Sciences 2025, 15, 7798. https://doi.org/10.3390/app15147798

## 개요

제조 공정은 여러 작업(task)이 방향성 비순환 그래프(DAG)로 연결된 구조로 모델링됩니다. 각 작업은 입력 버퍼와 사이클 타임에 따라 동작하며, 특정 시점 이후 병목이 발생할 작업을 예측하는 것이 목표입니다.

![Low information gain split ratio](assets/paper-figure-001.png)

논문은 Random Forest의 많은 split 중 일부만 높은 정보이득을 갖는다는 점에 주목합니다. MicroForest는 task별 Random Forest teacher에서 중요한 split rule만 추출하고, 그 rule pool을 이용해 작은 `MicroTree`를 구성합니다.

![Manufacturing DAG example](assets/paper-figure-002.png)

## 구현 내용

이 저장소에는 다음 요소가 포함되어 있습니다.

- DAG 기반 제조 공정 시뮬레이터
- 합성 병목 예측 데이터셋 생성기
- scikit-learn `RandomForestClassifier` 기반 teacher 모델
- Random Forest 내부 노드에서 정보이득 상위 split을 추출하는 `SplitSelector`
- 선택된 split pool만 탐색해 만드는 `MicroTree`
- task별 `MicroTree`를 묶은 `MicroForest`
- 전체 파이프라인 smoke test

![MicroForest overview](assets/paper-figure-003.png)

## 설치

Python 3.10 이상을 권장합니다. Anaconda를 사용 중이라면 별도 가상환경 없이도 아래처럼 바로 의존성을 설치할 수 있습니다.

```powershell
python -m pip install -r requirements.txt
```

일반 Windows Python에서 가상환경을 만들 경우:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

MSYS Python은 `numpy` wheel을 바로 사용하지 못해 소스 빌드로 빠질 수 있으므로 권장하지 않습니다.

## 빠른 확인

훈련을 오래 돌리지 않고, 데이터 생성부터 모델 저장까지 정상 동작하는지만 확인합니다.

```powershell
python scripts/smoke_test.py
```

성공하면 다음 정보가 출력됩니다.

- 생성된 task, edge, feature 수
- 학습 시간
- macro precision / recall / F1
- teacher RF node 수
- MicroTree node 수
- 저장된 dataset 및 model 경로

## 데이터 생성

```powershell
python scripts/generate_dataset.py --tasks 20 --samples 500 --out data/sample.csv
```

생성되는 CSV는 feature column 뒤에 `y_task_0 ... y_task_N` 형식의 label column을 포함합니다.

## MicroForest 학습

```powershell
python scripts/train_microforest.py --data data/sample.csv --tasks 20 --model-out artifacts/microforest.pkl
```

기본값은 빠른 확인을 위해 작게 잡혀 있습니다. 논문 규모의 실험을 재현하려면 task 수, sample 수, teacher estimator 수, depth 등을 늘려야 합니다.

## 코드 구조

```text
microforest/
  data.py          CSV 저장 및 로딩
  metrics.py       binary / macro metric 계산
  models.py        SplitSelector, MicroTree, MicroForest
  simulation.py    DAG 제조 공정 시뮬레이터
scripts/
  generate_dataset.py
  smoke_test.py
  train_microforest.py
tests/
  test_pipeline.py
assets/
  paper-figure-001.png ... paper-figure-008.png
```

## 현재 범위

현재 코드는 다음을 목표로 합니다.

- 논문의 핵심 모델링 아이디어를 읽을 수 있는 형태로 재구현
- 데이터 생성부터 모델 동작 확인까지 한 번에 실행 가능
- GitHub에 공개 가능한 최소 연구 아티팩트 제공

아직 포함하지 않은 항목은 다음과 같습니다.

- LightGBM, DCT, SSF 등 모든 baseline 비교
- 논문 표의 전체 실험 재현
- 실제 edge device latency 측정
- task grouping 최적화 실험

## Citation

```bibtex
@article{yoo2025microforest,
  title = {MicroForest: Lightweight Bottleneck Prediction for Manufacturing Processes on Edge Devices},
  author = {Yoo, Seungmin and Oh, Chanyoung},
  journal = {Applied Sciences},
  volume = {15},
  number = {14},
  pages = {7798},
  year = {2025},
  doi = {10.3390/app15147798}
}
```
