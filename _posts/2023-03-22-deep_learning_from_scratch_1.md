---
title:  "[Study][밑바닥딥러닝1] Ⅰ. Chapter 1 헬로 파이썬"
excerpt: "Deep Learning from Scratch 1, Ch1"

categories:
  - Study
tags:
  - [Python, DeepLearning, Korean]

toc: true
toc_sticky: true

date: 2023-03-22 20:04:49 +0900
last_modified_at: 2023-03-22 20:04:49 +0900
---
부산대학교 정보컴퓨터공학부 AID 동아리, 스터디 '밑바닥부터 시작하는 딥러닝 1권 (홀수팀)' 스터디 관련 질문들과, 추가로 공부하면 좋을 내용을 기록합니다.

주 교재 : 사이토 고키 저, 밑바닥부터 시작하는 딥러닝 ([링크](https://search.shopping.naver.com/book/catalog/32486532054?cat_id=50010921&frm=PBOKMOD&query=%EB%B0%91%EB%B0%94%EB%8B%A5%EB%B6%80%ED%84%B0+%EC%8B%9C%EC%9E%91%ED%95%98%EB%8A%94+%EB%94%A5%EB%9F%AC%EB%8B%9D&NaPm=ct%3Dlfjfev00%7Cci%3D63d2cbc6e28f9f3a3e6f6caff1ad43becd7611d1%7Ctr%3Dboknx%7Csn%3D95694%7Chk%3D6658236756ea9ddff6f3427c3aea96229d588096))

---

## Chapter 1. 헬로 파이썬

Chapter 1에서는 Python 설치와 Anaconda 설치를 돕고, 간단한 파이썬 문법과 numpy.array, matplotlib에 대해 소개한다.

필자가 현재 사용하는 Python과 conda 버전은 `conda 23.1.1, Python 3.11.0`이다. 작성일 기준으로, 최신 라이브러리를 사용한다. (사용하려고 항상 노력한다.)

파이썬의 기본적인 내용을 다루기 위한 스터디는 아니므로, 이번 Chapter에서는 numpy나 matplotlib와 관련된 내용보다는, 개인적으로 느꼈던 Python 문법의 사소한 꿀팁과, ReGex 정도만 서술하고 넘어가려 한다.

---

### - 산술 연산

너무나도 당연한 이야기이겠지만, 파이썬은 아주 간단하게 산술 연산을 지원한다.

```python
>>> (1 + 2) + (7 - 6)
4
>>> (5 * 20) / (2 ** 2)
25.0
```

그러나, 일반적인 수학적 방식과 항상 동일하게 동작하는 것이 아님에 유의하여야 한다.

다음과 같은 예시를 들 수 있다.

```python
>>> 10**12345 / 6

Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
OverflowError: integer division result too large for a float
```

Python도, (너무 당연하겠지만,) float로 IEEE 754 배정밀도 표현 방식을 사용하고 있고, 따라서 float로 표현될 수 있는 범위는 당연히 제한된다.

개인적으로 큰 수를 다룰 때, 상당히 많이 하였던 실수였던 것 같다.

---

### - match-case

`match-case` 문은 3.10 버전에 추가된, 아주 따끈따끈한 문법이다.

아직 [Python 공식 Reference](https://docs.python.org/ko/3/reference/compound_stmts.html#the-match-statement)에서도 한국어로는 번역되지 않았을 만큼, 생소한 사람도 많을 것이라 생각된다.

용례는 C의 `switch-case`문과 비슷한 방법으로, 다음과 같다.

```python
match status:
    case 400:
        return "Bad request"
    case 404:
        return "Not found"
    case 418:
        return "I'm a teapot"
    case _:
        return "Something's wrong with the internet"
```

안타깝게도, C 계열의 `switch-case`가 jump table을 만들어 효율적으로 프로그램을 동작할 수 있는 것과는 다르게, Python의 `match-case`는 단순히 if-elif-else를 대신하는 것이라 그 효용은 조금 떨어진다는 단점이 있지만, 코드의 길이를 줄여 가독성을 높이는 데에는 장점이 있을 것이다.

---

### - 정규표현식

상당히 많은 경우에, 어떠한 문자열을 처리하기 위해 정규표현식을 자주 사용한다. 이 책에서는 문자열을 처리하는 과정이 크게 들어가 있지 않아, 중요한 부분은 아닐 수 있으나 앞으로 문자열 처리와 관련된 사항이 생길 때, 유용하게 사용될 것이다.

```python
import re
p = re.compile('ab*')

print(type(p)) # print <class 're.Pattern'>
```

파이썬은 정규 표현식을 지원하기 위해, `re` (통칭 RegEx, Regular Expression의 약어) 모듈을 제공한다. `compile()`을 통해 정규식을 컴파일하면, '패턴 객체'로 컴파일된다.

``. ^ $ * + ? { } [ ] \ | ( )``

정규 표현식은 위의 메타 문자와 함께 사용된다. 위의 메타 문자와 관련하여, 사용될 수 있는 용례는 무궁무진하지만, 모든 용례를 다 설명할 수는 없으므로 메타 문자 [ ], ., *, + 정도만 간략하게 설명하려 한다.

해당 문자 이외의, 수많은 용례에 대해서는 [A.M.Kuchling 저자의 정규식 HOWTO](https://docs.python.org/ko/3/howto/regex.html)를 참고하면 많은 도움이 될 것이다.

#### ■ 문자 클래스 [ ]

가장 먼저 살펴볼 것은 문자 클래스 `[ ]`이다. 문자 클래스로 만들어진 정규식은 `"[ ] 사이의 문자들과 일치"`라는 의미를 갖는다.

문자는 `[abc] (a, b, c 각각과 일치)`와 같이 나열되거나, `[a-c]`와 같이 하이픈을 사용해 범위로 표현될 수 있다. (`[abc]`와 일치한다.)

문자 클래스 내부에서는, 어떤 문자나 메타 문자도 사용될 수 있다.

#### ■ 모든 문자 . (Dot)

두 번째로 살펴볼 것은, `.`(Dot)이다. Dot은 (줄바꿈 문자 \n을 제외한) `모든 문자 1개와 일치`라는 의미를 갖는다.

`a.c`와 같이 사용되었다면, `a + 모든 문자 + b`와 일치한다.

`a[.]c`와 같이 `[]` 사이에서 `.`을 사용할 경우, 문자 원래의 의미인 마침표가 된다.

#### ■ 반복 * (Asterisk), +(Plus)

세 번째로 살펴볼 것은, `*` (Asterisk)이다. Asterisk는 `* 바로 앞에 있는 문자가 0부터 무한대로 반복될 수 있음`이라는 의미를 갖는다.

`do*g`와 같이 사용되었다면, `dg`, `dog`, `doooooooooog` 등과 일치한다.

`+` (Plus)도 비슷한 의미를 가진다. Plus는 `+ 바로 앞에 있는 문자가 1부터 무한대로 반복될 수 있음`이라는 의미를 갖는다.

`ca+t`와 같이 사용되었다면, `cat`, `caaaaaaaaaaaaaaaat` 등과 일치한다.
