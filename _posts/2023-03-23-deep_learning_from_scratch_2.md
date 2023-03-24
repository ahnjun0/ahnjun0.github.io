---
title:  "[Study][밑바닥딥러닝1] Ⅱ. Chapter 2 퍼셉트론"
excerpt: "Deep Learning from Scratch 1, Ch2"

categories:
  - Study
tags:
  - [Python, DeepLearning, Korean]

toc: true
toc_sticky: true
math: true

date: 2023-03-23 01:26:49 +0900
last_modified_at: 2023-03-24 00:17:26 +0900
---
부산대학교 정보컴퓨터공학부 AID 동아리, 스터디 '밑바닥부터 시작하는 딥러닝 1권 (홀수팀)' 스터디 관련 질문들과, 추가로 공부하면 좋을 내용을 기록합니다.

주 교재 : 사이토 고키 저, 밑바닥부터 시작하는 딥러닝 ([링크](https://search.shopping.naver.com/book/catalog/32486532054?cat_id=50010921&frm=PBOKMOD&query=%EB%B0%91%EB%B0%94%EB%8B%A5%EB%B6%80%ED%84%B0+%EC%8B%9C%EC%9E%91%ED%95%98%EB%8A%94+%EB%94%A5%EB%9F%AC%EB%8B%9D&NaPm=ct%3Dlfjfev00%7Cci%3D63d2cbc6e28f9f3a3e6f6caff1ad43becd7611d1%7Ctr%3Dboknx%7Csn%3D95694%7Chk%3D6658236756ea9ddff6f3427c3aea96229d588096))

---

## Chapter 2. 퍼셉트론

퍼셉트론(Perceptron)은 다수의 신호를 입력으로 받아 하나의 신호를 출력하는 알고리즘이다. (논리회로및설계 시간에 배웠던 Multiplexer가 순간 생각났다.)

![그림 2-1. 입력이 n개인 퍼셉트론](/assets/img/2023/SCRATCH_1%20study/ch2/2-1.png){: width="250px"}_그림 2-1. 입력이 $n$개인 퍼셉트론_

[그림 2.1]은 입력으로 $n$개의 신호를 받은 퍼셉트론의 예다. $x$는 입력 신호, $y$는 출력 신호, $w$는 가중치를 뜻한다 (weight의 머리글자). 그림의 원은, **뉴런** 혹은 **노드**라고 부른다.

입력 신호가 뉴런에 보내질 때는, 각각 고유한 **가중치**가 곱해진다($w_1x_1, w_2x_2, ...$). 뉴런에서 보내온 신호의 총합이 정해진 한계를 넘어설 때만, 1을 출력한다. ('뉴런이 활성화한다' 라고도 표현한다.) 그 한계를 **임계값**이라고 부르고, $\theta$ 기호로 나타낸다.

퍼셉트론은 복수의 입력 신호 각각에 고유한 가중치를 부여한다. 가중치가 클수록 해당 신호가 그만큼 더 중요함을 뜻한다.

---

### 퍼셉트론 구현하기

우선, 입력이 두 개인 퍼셉트론을 구현해보자.

퍼셉트론의 동작을 수식으로 나타내면 다음과 같다.

$y = \begin{cases} 0 \ (b+w_1x_1 + w_2x_2 \leq \theta) \\\\ 1 \ (b+w_1x_1 + w_2x_2 > \theta) \end{cases}$

이 때, $b$를 편향(bias)라고 한다. 편향은 뉴런이 얼마나 쉽게 활성화되는지를 결정한다.

---

#### AND(논리곱, $\land$) 게이트

AND 게이트의 진리표(Truth Table)는 다음과 같다.

| $x_1$ | $x_2$ | $y$ |
| ------- | ------- | ----- |
| 0       | 0       | 0     |
| 0       | 1       | 0     |
| 1       | 0       | 0     |
| 1       | 1       | 1     |

따라서, 가중치와 편향을 도입한 AND 게이트는 다음과 같이 구현할 수 있다.

```python
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])  # Weight, 가중치
    b = -0.7  # Bias, 편향
    return 1 if (np.sum(x * w) + b) > 0 else 0
```

진리표와 같이, $x_1$이나 $x_2$ 둘 중 하나만 1이 된다면 편향에 의해 퍼셉트론에서 출력되는 값은 0이 되지만, $x_1$과 $x_2$ 둘 다 1이 된다면 함수에서 출력되는 값은 1이 된다.

#### OR (논리합, $\lor$)게이트

OR 게이트의 진리표(Truth Table)는 다음과 같다.

| $x_1$ | $x_2$ | $y$ |
| ------- | ------- | ----- |
| 0       | 0       | 0     |
| 0       | 1       | 1     |
| 1       | 0       | 1     |
| 1       | 1       | 1     |

따라서, 가중치와 편향을 도입한 OR 게이트는 다음과 같이 구현할 수 있다.

```python
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])  # Weight
    b = -0.2  # Bias, 편향
    return 1 if (np.sum(x * w) + b) > 0 else 0
```

진리표와 같이, $x_1$이나 $x_2$ 둘 중 하나 이상이 1이 된다면 퍼셉트론에서 출력되는 값은 1이 된다.

#### NAND (부정 논리곱, $\bar\land$)게이트

NAND 게이트의 진리표(Truth Table)는 다음과 같다.

| $x_1$ | $x_2$ | $y$ |
| ------- | ------- | ----- |
| 0       | 0       | 1     |
| 0       | 1       | 1     |
| 1       | 0       | 1     |
| 1       | 1       | 0     |

NAND 게이트는 NOT + AND 게이트로, AND 게이트를 응용하면 다음과 같이, 편향과 가중치의 부호를 뒤집어주는 것 만으로 쉽게 구현할 수 있다.

```python
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])  # Weight
    b = 0.7  # Bias, 편향
    return 1 if (np.sum(x * w) + b) > 0 else 0
```

진리표와 같이, $x_1$이나 $x_2$ 둘 다 1이 되지 않는다면 퍼셉트론에서 출력되는 값은 1이 된다.

#### NOR (부정 논리합, $\bar\lor$)게이트

NOR 게이트의 진리표(Truth Table)는 다음과 같다.

| $x_1$ | $x_2$ | $y$ |
| ------- | ------- | ----- |
| 0       | 0       | 1     |
| 0       | 1       | 0     |
| 1       | 0       | 0     |
| 1       | 1       | 0     |

NOR 게이트는 NOT + OR 게이트로, OR 게이트를 응용하면 다음과 같이, 편향과 가중치의 부호를 뒤집어주는 것 만으로 쉽게 구현할 수 있다.

```python
def NOR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])  # Weight
    b = 0.2  # Bias, 편향
    return 1 if (np.sum(x * w) + b) > 0 else 0
```

진리표와 같이, $x_1$이나 $x_2$ 둘 중 어느 것이라도 1이 되지 않는다면 퍼셉트론에서 출력되는 값은 1이 된다.

#### XOR (배타적 논리합, $\veebar$)게이트

XOR 게이트의 진리표(Truth Table)는 다음과 같다.

| $x_1$ | $x_2$ | $y$ |
| ------- | ------- | ----- |
| 0       | 0       | 0     |
| 0       | 1       | 1     |
| 1       | 0       | 1     |
| 1       | 1       | 0     |

XOR(eXclusive-OR)

```python
def XOR(x1, x2):
    return AND(NAND(x1, x2), OR(x1, x2))
```
