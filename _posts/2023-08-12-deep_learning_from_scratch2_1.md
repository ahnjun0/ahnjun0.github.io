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

신경망의 추론이나 학습에서는 다수의 샘플 데이터(**미니배치**, Minibatch)를 한꺼번에 처리합니다. 이렇게 하려면, 행렬 $\mathrm{x}$의 행 각가에 샘플 데이터를 하나씩 저장해야 합니다. 마치 다음 그림처럼요.

![형상 확인](/assets/img/2023/SCRATCH_2 study/ch1/fig 1-9.png){: width="400px"}

위의 그림과 같이 형상 확인을 통해 각 미니배치가 올비르게 변환되었는지를 알 수 있습니다.

완전연결계층에 의한 변환의 미니배치 버전을 구현하면 다음과 같습니다.

```python
W1 = np.random.randn(2, 4) # 가중치(weight)
b1 = np.random.randn(4) # 편향(bias)
x = np.random.randn(10, 2) # 입력
h = np.matmul(x, W1) + b1
```

이 예에서는 10개의 샘플 데이터 각각을 완전연결계층으로 변환합니다. 이때 x의 첫 번째 차원이 각 샘플 데이터에 해당됩니다. 또한, 마지막 b1의 덧셈은 브로드캐스트됩니다.

그런데 완전연결계층에 의한 변환은 '선형 변환'입니다. 여기에 '비선형 효과'를 부여해 표현력을 높이는 것이 바로 활성화 함수입니다.

여기서는 시그모이드 함수(Sigmoid Function)를 활성화 함수로 사용하였습니다. 시그모이드 함수는 다음과 같습니다.

$$ \sigma (x) =  \frac{\mathrm{1}}{\mathrm{1} + exp(-x)} $$

시그모이드 함수를 이용하여 비선형 변환을 하고, 신경망의 추론을 구현하는 파이썬 코드는 다음과 같습니다.

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.random.randn(10,2)
W1 = np.random.randn(2,4)
b1 = np.random.randn(4)
W2 = np.random.randn(4,3)
b2 = np.random.randn(3)

h = np.matmul(x, W1) + b1
a = sigmoid(h) # 활성화_Activation
s = np.matmul(a, W2) + b2
```

x의 형상(shape)은 (10,2)이고, s의 형상은 (10,3)입니다. 이는 10개의 데이터가 한 번에 처리되었고, 각 데이터는 3차원 데이터로 변환되었다는 뜻입니다.

그런데 이 신경망은 3차원 데이터를 출력합니다. 따라서 각 차원의 값을 이용하여 3-클래스 분류를 할 수 있고, 각 차원은 각 클래스에 대응하는 점수(score)가 됩니다.

즉, 실제로 분류를 한다면 출력층에서 가장 큰 값을 내뱉는 뉴런에 해당하는 클래스가 예측 결과가 되는 것입니다.

### 1.2.2 계층으로 클래스화 및 순전파 구현

신경망에서 하는 처리를 계층(layer)로 구현해 봅시다.

완전연결계층에 의한 변환을 Affine 계층, 시그모이드 함수에 의한 변환을 Sigmoid 계층으로 구현하였습니다.

또한 기본 변환을 수행하는 메서드(**순전파**)의 이름은 forward()로 하였습니다.

> Affine 계층이란?
> 유클리드 기하학의 Affine 변환([Wikipedia](https://en.wikipedia.org/wiki/Affine_transformation))을 수행하는 계층입니다.
> Affine 변환이란 직선과 평형성을 그대로 유지하는 기하 변환입니다. 즉, 어떠한 벡터를 선형 변환한 후 평행 이동시킨 것이라 볼 수 있습니다.

교재에서 제공하는 다음 '구현 규칙'을 따라 구현해봤습니다.

- 모든 계층은 forward() _(순전파)_ 와 backward() _(역전파)_ 메서드를 가진다.
- 모든 계층은 인스턴스 변수인 params와 grads를 가진다.

위의 구현 규칙에 따라, Sigmoid 계층과 Affine 계층을 구현해 보았습니다.

```python
class Sigmoid:
    def __init__(self) -> None:
        self.params = []
    
    def forward(self, x):
        return 1 / (1 + np.exp(-x))
    
class Affine:
    def __init__(self, W, b) -> None:
        self.params = [W, b]
    
    def forward(self, x):
        W, b = self.params
        return np.matmul(x, W) + b
```

이 때, Sigmoid 계층에는 매개변수가 따로 없으므로 params는 빈 리스트로 초기화하고, Affine 계층은 가중치와 편향의 영향을 받으므로 params에 보관하고, forward(x)에서 순전파 처리를 구현합니다.

위에서 구현한 두 계층을 사용하고, 아래의 같이 구성된 신경망의 추론을 구현해 보겠습니다.

![형상 확인](/assets/img/2023/SCRATCH_2 study/ch1/fig 1-11.png){: width="400px"}

```python
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size) -> None:
        I, H, O = input_size, hidden_size, output_size
        
        W1 = np.random.randn(I, H)
        b1 = np.random.randn(H)
        W2 = np.random.randn(H, O)
        b2 = np.random.randn(O)
        
        # 계층 생성
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]
        
        self.params = []
        for layer in self.layers:
            self.params += layer.params
        
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x


## 추론 수행

x = np.random.randn(10, 2)
model = TwoLayerNet(2, 4, 3)
s = model.predict
```

이상으로 입력 데이터 x에 대한 점수(s)를 구할 수 있습니다.

## 1.3 신경망의 학습

신경망이 추론을 수행할 때, 학습을 먼저 수행하고 그 학습된 매개변수를 이용해 추론을 수행하여야 합니다.

한편, 신경망의 학습은 **최적의 매개변수 값을 찾는 과정**입니다.

### 1.3.1 손실 함수

신경망 학습이 제대로 되어 가고 있는지 알기 위한 '척도'가 필수입니다. 수학 공부할 때 답지 없이 문제를 계속 풀다 보면, 이상한(?) 방향으로 학습이 이루어질 수 있는 것과 비슷합니다.

학습 단계의 특정 시점에서 신경망의성능을 나타내는 척도로 손실(loss)를 사용합니다. 손실은 학습 데이터와 신경망 예측 결과를 비교하여 예측이 얼마나 나쁜가를 나타내는 스칼라값입니다.

신경망의 손실은 **손실 함수**를 이용해 구합니다.

손실 함수의 종류는 MSE, MAE, Entropy, Cross-Entropy, Binary Crossentropy 등으로 굉장히 다양한데 해당 교재에서는 다중 클래스 분류 신경망에서 사용하는 손실 함수로 교차 엔트로피 오차(Cross-Entropy Error)를 사용하였습니다.

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
