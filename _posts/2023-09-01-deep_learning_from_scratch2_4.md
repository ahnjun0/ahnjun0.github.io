---
title:  "[Study][밑바닥딥러닝2] Ⅳ. Chapter 4 word2vec 속도 개선"
excerpt: "Deep Learning from Scratch 2, Ch1"

categories:
  - Study
tags:
  - [Python, DeepLearning, Korean]

toc: true
toc_sticky: true
math: true

date: 2023-09-01 20:00:00 +0900
last_modified_at: 2023-09-01 20:00:00 +0900
---
부산대학교 정보컴퓨터공학부 AID 인공지능 동아리, 스터디 '밑바닥부터 시작하는 딥러닝 2권' 스터디 관련 질문들과, 추가로 공부하면 좋을 내용을 기록합니다.

블로그에 사용되는 교재의 그림, 수식 등은 [옮긴이 개앞맵시님 repo](https://github.com/WegraLee/deep-learning-from-scratch-2)에서 가져왔습니다.

주 교재 : 사이토 고키 저, 밑바닥부터 시작하는 딥러닝 2 ([링크](https://search.shopping.naver.com/book/catalog/32482740788?cat_id=50010921&frm=PBOKPRO&query=%EB%B0%91%EB%B0%94%EB%8B%A5%EB%B6%80%ED%84%B0+%EC%8B%9C%EC%9E%91%ED%95%98%EB%8A%94+%EB%94%A5%EB%9F%AC%EB%8B%9D+2&NaPm=ct%3Dll7ydl08%7Cci%3Dd83faa3f1dfc4602293fd85f5208fc9634129e3c%7Ctr%3Dboknx%7Csn%3D95694%7Chk%3Dfbe1bddbc75c01bb64daf8d7c2894fe0ff606559))

---

## Chapter 4. word2vec 속도 개선

3장에서 word2vec에 대해 학습하고, 여러 단어로부터 하나의 단어를 추측하는 CBOW 모델과 skip-gram 모델을 구현했습니다.

그러나 두 모델은 어휘 수가 많아지면, 계산량도 동일하게 커져 시간이 너무 오래 걸리는 문제가 발생합니다. CBOW 모델과 skip-gram 모델에서는 Softmax-with-loss 계층을 거쳐 단어 집합 크기의 벡터와 원-핫 벡터와의 오차를 계산하여 가중치 값이 수정되는 일련의 작업이 이루어지는데, 단어 집합 $V$가 어느 정도를 넘어서면 계산량이 너무 커져 계산 시간이 너무 오래 걸릴 뿐만 아니라, 상당한 메모리를 차치할 수 밖에 없기 때문이죠.

따라서, 이번 4장에서는 word2vec의 속도를 개선할 수 있는 여러가지 방법을 소개합니다. 책에서는 1. Embedding이라는 새 계층을 도입하는 것과, 2. Negative Sampling을 이용하는 두 가지 방법을 소개하고 있고, 저는 거기에 더해 Hierarchical Softmax을 이용하는 방법을 소개하겠습니다.

<!--
[NLP | TIL] Negative Sampling과 Hierarchical Softmax, Distributed Representation 그리고 n-gram
: https://velog.io/@xuio/NLP-TIL-Negative-Sampling%EA%B3%BC-Hierarchical-Softmax-Distributed-Representation-%EA%B7%B8%EB%A6%AC%EA%B3%A0-n-gram -->

<!-- 
계층적 소프트맥스(Hierarchical Softmax, HS) in word2vec
https://uponthesky.tistory.com/15 -->

<!-- Keras Embedding은 word2vec이 아니다
https://heegyukim.medium.com/keras-embedding%EC%9D%80-word2vec%EC%9D%B4-%EC%95%84%EB%8B%88%EB%8B%A4-619bd683ded6 -->

<!-- [NLP] Word2Vec: (2) CBOW 개념 및 원리
https://heytech.tistory.com/352 -->

위의 방법으로 '진짜' word2vec이 완성되면, 해당 데이터셋을 가지고 학습을 수행해보겠습니다.

## 4.1 word2vec 개선 ①

Embedding 계층을 도입하여, 입력층의 원-핫 표현과 가중치 행렬 $W_{in}$의 곱을 계산할 때의 문제를 해결해 봅시다.

### 4.1.1 Embedding 계층 / 4.1.2 Embedding 계층 구현

앞 장의 word2vec 구현에서는, 단어를 원-핫 표현으로 바꾸고, 그것을 MatMul 계층에서 가중치 행렬을 곱했습니다. 그림으로 표현하면 아래와 같죠.

![앞 장에서의 word2vec 구현](/assets/img/2023/SCRATCH_2 study/ch4/fig 4-3.png){: width="400px"}

위의 그림에서 결과적으로 수행하는 일이 무엇인가요?

단지 행렬의 특정 행을 추출하는 것 뿐입니다. 원-핫 표현으로 만들고, 행렬곱을 계산하는 일은 사실은 불필요한 일일 뿐이죠.

따라서, 가중치 매개변수로부터 '단어 ID에 해당하는 벡터를 추출하는 계층'을 만들면 해당 계산을 간단하게 줄일 수 있습니다. 해당 계층을 Embedding 계층으로 정합니다.

설명을 어렵게 했지만, 행렬에서 특정 행을 추출하기란 아주 쉽습니다.

다음과 같이 말이죠.

```python
W = np.array([[0,1,2],
               [3,4,5],
               [6,7,8]])

print(W[2])
# [6 7 8]
```

그렇다면, 위의 과정을 수행하는 Embedding 계층의 forward(), backward() 메서드를 구현해 봅시다.

```python
class Embedding:
    def __init__(self, W) -> None:
        self.params = [W]
        self.grads  = [np.zeros_like(W)]
        self.idx = None
        
    def forward(self, idx):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out
    
    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0
        
        np.add.at(dW, self.idx, dout)

        # same as..
        # for i, wordID in enumerate(self.idx):
        #     dW[wordID] += dout[i]
            
        return None

```

> dW[...] = 0에서는 dW의 형상을 모듀 유지한 채, 그 원소를 모두 0으로 만듭니다.
>
> '...'을 ellipsis라고 하는데, (2D array일 경우에) 모든 행과 열을 선택하는 dW[:,:]과 같은 역할을 합니다.

idx의 원소가 중복되었을 때를 대비해서, dW에 값을 '할당'하는 방식이 아니라, np.add.at()문을 이용하여 값을 dW의 해당 인덱스에 '**더하는**' 방식으로 진행됩니다.

이상으로 Embedding 계층을 구현하여, Matmul 계층을 Embedded 계층으로 전환할 수 있게 되었습니다.

## 4.2 word2vec 개선 ②

앞에서 이야기했듯, 이번에 개선할 것은 은닉층 이후의 처리인, 행렬곱과 Softmax 계층의 계산에서 병목을 해결하는 것이 목표입니다. 이 때, **네거티브 샘플링** 기법을 사용합니다. Softmax 대신 네거티브 샘플링을 이용하면 어휘가 아무리 많아져도 일정하게 계산량을 낮을 수준에서 억제할 수 있습니다.

### 4.2.1 은닉층 이후 계산의 문제점 / 4.2.2 다중 분류에서 이진 분류로

은닉층 이후에서 계산이 많이 걸리는 부분은, 두 부분으로 나눌 수 있습니다,

- 은닉층의 뉴런과 가중치 행렬(즉, $W_{out}$)의 곱
- Softmax 계층의 계산

우선, 첫 번째 문제는 거대한 두 행렬의 행렬곱을 구하는 문제입니다. 순전파 때와 역전파 때, 같은 계을 수행하기 때문에 이 행렬곱 계산을 가볍게 만드는 것은 무척이나 절실한 일입니다.

두 번째로, Softmax에서도 같은 문제가 발생합니다. 다음 Softmax의 식을 봅시다.

$$ y_k= \frac{exp(s_k)}{\sum_{i=1}^{N}exp(s_i)} $$

이 때, $N$은 어휘 수입니다. Softmax를 계산하기 위해선, 분모의 계산을 $N$번 수행하여야 분모의 값을 얻을 수 있으므로, 계산이 $N$번 필요합니다. 이 또한 $N$이 커지면 계산량이 비례하여 증가하므로 가벼운 계산이 절실합니다.

이를 해결하기 위해, **네거티브 샘플링 기법**을 사용합니다.

이때까지는,

### 4.2.3 시그모이드 함수와 교차 엔트로피 오차

### 4.2.4 다중 분류에서 이진 분류로(구현)

```python
class EmbeddingDot:
    def __init__(self, W) -> None:
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None
        
    def forward(self, h, idx):
        target_W = self.embed.forward(idx)
        out = np.sum(target_W * h, axis=1)
        
        self.cache = (h, target_W)
        return out
    
    def backward(self, dout):
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0], 1)
        
        dtarget_W = dout * h
        self.embed.backward(dtarget_W)
        dh = dout * target_W
        return dh
```

### 4.2.5 네거티브 샘플링

### 4.2.6 네거티브 샘플링의 샘플링 기법

### 4.2.7 네거티브 샘플링 구현

```python

class NegativeSamplingLoss:
    def __init__(self, W, corpus, power=0.75, sample_size=5) -> None:
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)]
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]
        self.params, self.grads = [], []
        
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads
    
    def forward(self, h, target):
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_sample(target)
        
        # Positive
        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype=np.int32)
        loss = self.loss_layers[0].forward(score, correct_label)
        
        # Negative
        negative_label = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size):
            negative_target = negative_sample[:, 1]
            score = self.embed_dot_layers[1+i].forward(h, negative_target)
            loss += self.loss_layers[1+i].forward(score, negative_label)
            
        return loss
        
    def backward(self, dout=1):
        dh = 0
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            dscore = l0.backward(dout)
            dh += l1.backward(dscore)
            
        return dh

```

## 4.3 개선된 word2vec 학습

### 4.3.1 CBOW 모델 구현 / 4.3.2 CBOW 모델 학습 코드

### 4.3.3 CBOW 모델 평가

## 4.4 word2vec 남은 주제

## A. Hierarchical Softmax
