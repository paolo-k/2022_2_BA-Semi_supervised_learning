# 2022_2_BA-Semi_supervised_learning
Tutorial Homework 5(Business Analytics class in Industrial &amp; Management Engineering, Korea Univ.)

# Overview of Semi-Supervised Learning (SSL)
## What is "Semi"-Supervised Learning
Supervised Learning은 y label이 있을 때 사용 가능하고, Unsupervised Learning은 label이 없을 때 사용 가능하다. 그렇다면 SSL은 언제 사용 가능할까? SSL은 소수의 y label을 학습하여 다수의 unlabelled dataset을 분류하는 방법이다. 예시로, 고양이라는 y label이 있는 사진이 있고, label이 없는 사진이 있을 때, 고양이 사진을 통해 뾰족한 귀, 수염, 큰 눈 등의 고양이의 특징을 학습한 모델은 비슷하게 생겼으나 label이 없는 다른 고양이 사진을 충분히 분류해낼 수 있을 것이다. 살다보면 언젠가는 Label이 일부 소수의 데이터셋에만 있고, 나머지 다수에는 없는 상황을 겪는다. 이 때, SSL은 그 진가를 발휘한다. 

![그림1](https://user-images.githubusercontent.com/106015570/209720224-e17729a6-2008-4437-b36f-17fd0f946a6d.png)

## Purpose of SSL

현실에서 Machine Learning(ML)이 필요한 과제에서, Labeled data는 유의미한 결과를 내기는 쉬우나, 얻기가 굉장히 어렵다. label을 부여할 장비가 필요하고, 맞는 label을 알고 있는 도메인 전문가가 필요하다. Unlabeled data는 반대로 얻기는 쉬우나, 유의미한 결과를 내기 어렵다. 데이터 특성을 활용한 군집 분석 및 분류 정도가 최대의 역할이다. 두 방법의 장, 단점을 절충하여 적은 양의 Labeled data를 효율적으로 활용하기 위하여 SSL이 탄생하였다.

![그림5](https://user-images.githubusercontent.com/106015570/209726604-94abe247-188f-417f-9c99-fbcbe6552291.png)

본 튜토리얼에서는 Street View Housing Number(SVHN) 데이터셋과, 이미 작성된 코드를 이용하여 iteration 100에서 Pseudo-Label, PI model, Mean teacher, Virtual Adversarial Training(VAT), MixMatch의 5가지 모형을 적용한 결과를 비교할 것이다.

# What is Pseudo-label, PI model, Mean teacher, VAT, MixMatch
## Pseudo-label

Pseudo-label은 supervised learning을 통한 prediction에 기반하여 threshold 등 간단한 규칙을 통해 unlabeled data에 pseudo label을 학습한 후, labeled data, pseudo-labeled data를 이용하여 모델을 다시 학습하는 것이다. SSL계의 고전인 만큼 굉장히 직관적이고 간단한 방법이다.

## PI model

PI model이란, augmentation한 input을 두 개 마련한 후, 다르게 dropout을 적용한 network에 각각의 augmentated set을 적용하여 도출된 결과에 따라 모델을 평가하는 방법이다. 즉, 같은 데이터셋 및 네트워크를 기반으로 다르게 가공한 두 개의 데이터셋 및 네트워크의 결과가 얼마나 일관되었는지를 통해 성능을 평가하는 consistency regularization 방법 중 하나이다.

## Mean teacher

Mean teacher 또한 PI model 처럼 consistency regularization 방법 중 하나로, 이름에 맞게 teacher model, student model로 나누어 각각의 모델에 노이즈를 부여한 뒤, 각각의 모델의 결과를 통해 성능을 평가한다. 이후, student 모델의 weight를 exponential moving average하여 teacher model의 weight를 업데이트한다.

## VAT

VAT도 consistency regularization에 속하는 방법이긴 하나, PI model, Mean teacher와는 접근이 다소 다르다. 결과의 일관성(consistency)을 보는 PI model, Mean teacher와 다르게, Unlabeled data의 가상의 적대적 방향(Virtual Adversarial)을 정의하여 학습시킨다. 즉, 분류경계면에 가까워 약간의 변형만으로 label이 완전히 바뀌어 부여되는 객체들을 분류경계면에서 멀어지도록 학습시키는 것이다.

## MixMatch

MixMatch는 위의 기법들을 포함하여, 효과가 좋은 SSL 기법들을 하나의 framework로 통합하는 Holistic Method이다. 가장 먼저 augmentation 된 Labeled data, Unlabeled data에 대하여 Unlabeled data에 대하여 Entropy를 최소화하도록 Guessed label을 부여한다. 그 다음, Labeled data, Unlabeled data를 모두 Mixup하여 모델을 학습시킨 후, Loss를 평가한다.

# Tutorial of SSL
## 코드 및 데이터 개요

본 tutorial은 다양한 SSL 기법에 대하여 결과를 도출하도록 기작성된 코드를 그대로 활용하였다. 레포지토리 URL은 아래와 같다.

https://github.com/perrying/realistic-ssl-evaluation-pytorch

본 tutorial에서는 각 건물의 번지 수를 활용한 SVHN 데이터셋을 활용하였다. Training set은 Labeled data 1,000개, Unlabeled data 64,931개로 도합 65,931개이며, Validation set은 7,326개, Testing set은 26,032개이다. 

![그림3](https://user-images.githubusercontent.com/106015570/209720284-7a1caa04-5f58-43fb-b281-5d7256bc5cf5.png)


## 결과 비교

연산 시간 이슈로 인하여, iteration 100을 적용하였다. 5개 모델을 비교한 결과는 아래와 같다.

|method|Supervised Loss|Unsupervised Loss|Time(iter/sec)|
|------|---|---|---|
|Pseudo Labelling|1.043|7.27E-03|0.602|
|Pi model|0.980|2.74E-05|0.600|
|Mean teacher|1.049|1.01E-05|0.505|
|VAT|1.195|1.01E-05|0.442|
|MixMatch|1.110|1.92E-02|0.404|

결과를 간단히 요약하자면, Supervised loss, Unsupervised loss는 모두 Pi model에서 가장 좋은 성능을 보였다. 반면 연산 시간 측면에서는 MixMatch의 성능이 가장 좋았으며, 최신 모델일수록 더 좋은 성능을 보였다. Loss 측면에서 Pi model이 성능이 좋았던 이유를 추정하자면, 기본적으로 같은 데이터, 같은 네트워크에서 노이즈만 부여된 결과의 일관성을 이용하기에, 약간의 overfit이 발생한 것으로 보인다. 연산 시간 측면에서는, 기존의 모델이 한 epoch에서 update를 1회씩 수행하는 등 다소 비효율적으로 설계되었다는 점을 고려하면, 최신 모델일수록 효율성이 높은 것이 충분히 설명될 수 있다. 한 가지 흥미롭다고 생각되는 부분은, 의외로 저 중 가장 최신 방법인 MixMatch의 Unsupervised Loss 성능이 가장 낮다는 것이다. 이에 대해서는 모델 전체에서 Unsupervised Loss가 낮게 나온 것으로 보아, 데이터셋 자체가 상당히 맞추기 쉬운 편이므로 오래된 방법들도 충분히 좋은 성능을 발휘할 수 있었다는 점이 유력한 원인으로 추정된다.

모델들의 결과를 더 정확히 판단하기 위해서는 iteration을 충분히 하여, 안정적인 모델을 만들 필요가 있다.
