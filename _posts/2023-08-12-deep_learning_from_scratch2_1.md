---
title:  "[Study][밑바닥딥러닝2] Ⅰ. Chapter 1 신경망 복습"
excerpt: "Deep Learning from Scratch 2, Ch2"

categories:
  - Study
tags:
  - [Python, DeepLearning, Korean]

toc: true
toc_sticky: true
math: true

date: 2023-08-12 20:00:00 +0900
last_modified_at: 2023-03-22 20:00:00 +0900
---
부산대학교 정보컴퓨터공학부 AID 인공지능 동아리, 스터디 '밑바닥부터 시작하는 딥러닝 2권' 스터디 관련 질문들과, 추가로 공부하면 좋을 내용을 기록합니다.

블로그에 사용되는 교재의 그림, 수식 등은 [옮긴이 개앞맵시님 repo](https://github.com/WegraLee/deep-learning-from-scratch-2)에서 가져왔습니다.

주 교재 : 사이토 고키 저, 밑바닥부터 시작하는 딥러닝 2 ([링크](https://search.shopping.naver.com/book/catalog/32482740788?cat_id=50010921&frm=PBOKPRO&query=%EB%B0%91%EB%B0%94%EB%8B%A5%EB%B6%80%ED%84%B0+%EC%8B%9C%EC%9E%91%ED%95%98%EB%8A%94+%EB%94%A5%EB%9F%AC%EB%8B%9D+2&NaPm=ct%3Dll7ydl08%7Cci%3Dd83faa3f1dfc4602293fd85f5208fc9634129e3c%7Ctr%3Dboknx%7Csn%3D95694%7Chk%3Dfbe1bddbc75c01bb64daf8d7c2894fe0ff606559))

---

## Chapter 1. 신경망 복습을 들어가며

1권 내용을 완벽하게 블로그에 정리하지는 못했지만, 일단 2권 스터디를 진행하면서 2권 정리를 블로그에 진행해보고자 합니다.

## 1.1 수학과 파이썬 복습

밑바닥부터 시작하는 딥러닝 1권이나, 선형대수학 등의 과목에서 이미 많이 다루었던 부분이지만, 간단하게 정리해보았습니다.

### 1.1.1 벡터와 행렬

![벡터와 행렬의 예](/assets/img/2023/SCRATCH_2 study/ch1/fig 1-1.png){: width="400px"}

- **벡터**(Vector)는 크기와 방향을 가진 양이며, 파이썬에서는 '1차원' 배열로 취급할 수 있습니다.
  - 벡터는 열벡터와 행벡터 두 가지로 나뉩니다.
- **행렬**(Matrix)은 숫자가 '2차원' 형태로 늘어선 것입니다.
  - 가로줄을 '행'(row), 세로줄을 '열'(column)이라 합니다.

```python
# 벡터와 행렬 모두 numpy 모듈의 array() 메서드를 이용해서 나타낼 수 있다.
import numpy as np

# 벡터의 경우
x = np.array([1,2,3])
print(x.shape) # <- result : (3,) / 다차원 배열의 형상
print(x.ndim) # <- result : 1 / 차원 수(dimension)


# 행렬의 경우
W = np.array([[1,2,3],[4,5,6]]) # <- <class 'numpy.ndarray'>
print(W.shape) # <- result : (2,3) / 다차원 배열의 형상
print(W.ndim) # <- result : 2 / 차원 수(dimension)
```

### 1.1.2 행렬의 원소별 연산 / 1.1.3 브로드캐스트

numpy 배열은 더하기(+) 연산이나, 곱하기(*) 연산을 할 수 있습니다. 또한, 형상이 다른 배열끼리도 연산할 수 있습니다. 이를 **브로드캐스트**(Broadcast)라 합니다.
브로드캐스트는 몇 가지 규칙을 충족했을 때 효과적으로 동작합니다. [numpy.org에서 자세한 사항 확인하기](https://numpy.org/doc/stable/user/basics.broadcasting.html)

```python
W = np.array([[1,2,3],[4,5,6]])
x = np.array([[0,1,2],[3,4,5]])

print(W + x)
print(W * x)
# result : 
# [[ 1  3  5]
#  [ 7  9 11]]
# [[ 0  2  6]
#  [12 20 30]]

''' Broadcasting Example '''

A = np.array([1,2],[3,4])
B = np.array([10,20])

print(A * 10)
print(A * B)
# result : 
# [[10 20]
#  [30 40]]
# [[10 40]
#  [30 80]]
```

### 1.1.4 벡터의 내적과 행렬의 곱 / 1.1.5 행렬 형상 확인

벡터의 내적과 행렬의 곱 구하는 방법은, 선형대수학에서 익히 (지겹도록) 학습했을테니, 바로 그 방법만 코드로 확인하겠습니다.

```python
# 벡터의 내적(Dot Product) - np.dot
a = np.array([1,2,3])
b = np.array([4,5,6])

print(np.dot(a,b))
# result : 32

# 행렬의 곱(Matrix Multiplication) - np.matmul
A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])

print(np.matmul(A,B))
# result : 
# [[19 22]
#  [43 50]]
```

벡터의 내적과 행렬의 곱 모두에 np.dot()을 사용할 수 있다고는 합니다. 인수가 모두 1차원 배열이면 벡터의 내적을, 2차원 배열이면 행렬의 곱을 계산합니다. 다만, 교재에서는 헷갈리지 않도록 의도를 명확히 해서 둘을 구분하는 것을 권장하고 있습니다.

행렬을 계산할 때는 그 '형상'(shape)에 주의해야 합니다.
이 역시 수많은 선형대수학 수업에서 귀에 딱지가 앉도록 강조하는 내용이므로, 사진 한 장으로 갈음하고 넘어가겠습니다.

![형상 확인](/assets/img/2023/SCRATCH_2 study/ch1/fig 1-6.png){: width="400px"}

마지막으로 책에서는, numpy의 학습을 돕기 위해 '[100 Numpy Exercise](https://github.com/rougier/numpy-100)' 사이트를 추천하고 있습니다. 시간 날 때 한번 해보는걸로...

## 1.2 신경망의 추론

### 1.2.1 신경망 추론 전체 그림

여기 간단한 신경망의 예시를 하나 들죠.

![신경망의 예](/assets/img/2023/SCRATCH_2 study/ch1/fig 1-7.png){: width="400px"}

뉴런을 동그라미(○)로, 그 사이의 연결을 화살표(→)로 나타냈습니다. 이전에 학습하였듯, 뉴런 간의 연결에는 가중치(weight)와 편향(bias)가 존재합니다. 덧붙여, 그림의 신경망은 인접하는 층의 모든 뉴런과 연결되어 있다는 점에서 **완전연결계층**(fully connected layer)라고도 부릅니다.

그림의 신경망이 수행하는 계산을 수식으로 나타내면 다음과 같습니다.

$h_1 = x_1w_{11} + x_2w_{21} + b_1$

이 때, 입력층의 데이터는 $(x_1, x_2)$로 두고, 가중치는 $w_{11}$과 $w_{21}$로, 편향은 $b_1$로 하였습니다.

위 식과 같이 은닉층의 뉴런은 가중치의 합으로 계산됩니다. 이런 식으로 가중치와 편향의 값을 바꿔가며 식의 계산을 뉴런의 수만큼 반복하면 은닉층에 속한 모든 뉴런의 값을 구할 수 있습니다.

완전연결계층이 수행하는 변환은 행렬의 곱을 이용해 다음과 같은 식으로 간소화할 수 있습니다.

$\mathrm{h = xW + b}$

이때, $\mathrm{x}$는 입력, $\mathrm{h}$는 은닉층의 뉴런, $\mathrm{W}$는 가중치, $\mathrm{b}$는 편향을 뜻합니다. 이 때에도, 형상 확인은 물론 중요합니다.

### 1.2.2 계층으로 클래스화 및 순전파 구현

## 1.3 신경망의 학습

### 1.3.1 손실 함수

### 1.3.2 미분과 기울기

### 1.3.3 연쇄 법칙

### 1.3.4 계산 그래프

#### 곱셈 노드

#### 분기 노드

#### Repeat 노드

#### Sum 노드

#### Matmul 노드

### 1.3.5 기울기 도출과 역전파 과정

#### Sigmoid 계층

#### Affine 계층

#### Softmax with Loss 계층

### 1.3.6 가중치 갱신

## 1.4 신경망으로 문제를 풀다

### 1.4.1 스파이럴 데이터셋

### 1.4.2 신경망 구현

### 1.4.3 학습용 코드

### 1.4.4 Trainer 클래스

## 1.5 계산 고속화

### 1.5.1 비트 정밀도

### 1.5.2 GPU(쿠파이)

## 1.6 정리
