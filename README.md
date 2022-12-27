# 2022_2_BA-Semi_supervised_learning
Tutorial Homework 5(Business Analytics class in Industrial &amp; Management Engineering, Korea Univ.)

# Overview of Semi-Supervised Learning (SSL)
## What is "Semi"-Supervised Learning
Supervised Learning은 y label이 있을 때 사용 가능하고, Unsupervised Learning은 label이 없을 때 사용 가능하다. 그렇다면 SSL은 언제 사용 가능할까? SSL은 소수의 y label을 학습하여 다수의 unlabelled dataset을 분류하는 방법이다. 예시로, 고양이라는 y label이 있는 사진이 있고, label이 없는 사진이 있을 때, 고양이 사진을 통해 뾰족한 귀, 수염, 큰 눈 등의 고양이의 특징을 학습한 모델은 비슷하게 생겼으나 label이 없는 다른 고양이 사진을 충분히 분류해낼 수 있을 것이다. 살다보면 언젠가는 Label이 일부 소수의 데이터셋에만 있고, 나머지 다수에는 없는 상황을 겪는다. 이 때, SSL은 그 진가를 발휘한다. 

![그림1](https://user-images.githubusercontent.com/106015570/209720224-e17729a6-2008-4437-b36f-17fd0f946a6d.png)

## Purpose of SSL

현실에서 Machine Learning(ML)이 필요한 과제에서, Labeled data는 유의미한 결과를 내기는 쉬우나, 얻기가 굉장히 어렵다. label을 부여할 장비가 필요하고, 맞는 label을 알고 있는 도메인 전문가가 필요하다. Unlabeled data는 반대로 얻기는 쉬우나, 유의미한 결과를 내기 어렵다. 데이터 특성을 활용한 군집 분석 및 분류 정도가 최대의 역할이다. 두 방법의 장, 단점을 절충하여 적은 양의 Labeled data를 효율적으로 활용하기 위하여 SSL이 탄생하였다.

![그림4](https://user-images.githubusercontent.com/106015570/209721860-27aa776f-c7a8-4185-8856-3c507be7771f.png)

본 튜토리얼에서는 Street View Housing Number(SVHN) 데이터셋과, 이미 작성된 코드를 이용하여 iteration 100에서 Pseudo-Label, PI model, Mean teacher, Virtual Adversarial Training(VAT), MixMatch의 5가지 모형을 적용한 결과를 비교할 것이다.

![그림3](https://user-images.githubusercontent.com/106015570/209720284-7a1caa04-5f58-43fb-b281-5d7256bc5cf5.png)

# What is PI model, Mean teacher, VAT, Pseudo-Label, MixMatch
## Pseudo-Label



## PI model

PI model이란, augmentation한 input을 두 개 마련한 후, 다르게 dropout을 적용한 network에 각각의 augmentated set을 적용하여 도출된 결과에 따라 모델을 평가하는 방법이다. 즉, 같은 데이터셋 및 네트워크를 기반으로 다르게 가공한 두 개의 데이터셋 및 네트워크의 결과가 얼마나 일관되었는지를 통해 성능을 평가하는 consistency regularization 방법 중 하나이다.

## Mean teacher

Mean teacher 또한 PI model 처럼 consistency regularization 방법 중 하나로, 이름에 맞게 teacher model, student model로 나누어 각각의 모델에 노이즈를 부여한 뒤, 각각의 모델의 결과를 통해 성능을 평가한다. 이후, student 모델의 weight를 exponential moving average하여 teacher model의 weight를 업데이트한다.

## VAT

VAT도 consistency regularization에 속하는 방법이긴 하나, PI model, Mean teacher와는 접근이 다소 다르다. 결과의 일관성(consistency)을 보는 PI model, Mean teacher와 다르게, Unlabeled data의 가상의 적대적 방향(Virtual Adversarial)을 정의하여 학습시킨다. 즉, 분류경계면에 가까워 약간의 변형만으로 label이 완전히 바뀌어 부여되는 객체들을 분류경계면에서 멀어지도록 학습시키는 것이다.

