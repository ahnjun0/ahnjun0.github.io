---
title:  "2023 부산대학교 CodeRace 회고 + Python 문제풀이"
excerpt: "2023 부산대학교 CodeRace에 참여한 후, 개인적인 회고를 기록합니다."

categories:
  - Contest
tags:
  - [Contest, Korean, PNU]

toc: true
toc_sticky: true
math: true

date: 2023-07-20 20:30:41 +0900
last_modified_at: 2023-07-20 20:30:41 +0900
---
2023년 5월 6일에 개최된 부산대학교 Coding Contest인 CodeRace에 참여하고 나서, 내 문제풀이와 개인적인 소회를 적었습니다. 2022년 작년 대회에 이어, 올해 두 번째로 참여하는 대회인데 가면 갈수록 발전하는 대회인 것 같아 개인적으로 기분이 좋습니다.

[오픈 Contest in 백준 (링크)](https://www.acmicpc.net/contest/view/994)

결과부터 말하면, **Beginner** 대회에서 **은상(2등상)**을 수상하기'는' 했습니다.

![수상자 명단](/assets/img/2023/PNU_Coderace/award.jpg){: width="250px"}_수상자 명단_

사실, 수상을 했음에도, 대회 끝나고 이 내용을 바로 올리지 않은 이유는, (미루다 미루다 2개월이 훌쩍 지나서야 올린 이유는) 솔직히 말하자면 '과연 이걸 올려야 되나?' 이런 거 괜히 올리면 *'부산대 수준 겨우 이거?'*이런 소리 듣는 거 아냐? 라는 생각을 정말 많이 했었습니다. 뭐 그래도 좋은 게 좋은 거 아닐까요...? ㅎㅎ...

---

## 1번 문제 : 첨탑 밀어서 부수기

- Solved.ac 난이도 : Bronze 3
- 알고리즘 : 그리디 알고리즘
- [문제 보러가기](https://www.acmicpc.net/problem/28014)

1번 문제답게, 문제의 난이도는 그렇게 어렵지 않았습니다.

'밀려 넘어지는 첨탑의 높이가 바로 그다음 첨탑의 높이보다 클 때만 그다음 첨탑도 밀려 넘어진다.'라는 문장이 문제 전부를 관통하는 주제라고 생각했습니다. 즉, 여러 숫자가 순서대로 나열되어 있을 때, N번째 첨탑 <= N+1번째 첨탑일 때 밀어주는 횟수를 더하면 됩니다.

```python
import sys
input = lambda : sys.stdin.readline()

N = int(input())

top = list(map(int, input().split()))
push = 1

for i in range(N-1):
    if top[i] <= top[i+1]:
        push += 1

print(push)
```

밀어주는 횟수를 'push' 변수에 저장하고, list를 순회하며 'N번째 첨탑 <= N+1번째 첨탑' 조건을 만족하는 경우에, push 변수에 1을 더하는 것으로 하였습니다. 최초에 첨탑을 1회 밀어주어야 하므로, push는 1로 초기화하였습니다.

시험장에서는 가뿐히 풀고 넘어갈 줄 알았는데, '이상'과 '초과'를 혼동하는 바람에 아주 쉽게 넘어가지는 못했습니다. (문제를 꼼꼼히 읽는 습관을 들여야겠다...)

## 2번 문제 : 영역 색칠

- Solved.ac 난이도 : Silver 2
- 알고리즘 : 그리디 알고리즘
- [문제 보러가기](https://www.acmicpc.net/problem/28015)

2번 문제는,
