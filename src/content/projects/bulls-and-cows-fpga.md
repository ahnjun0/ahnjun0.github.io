---
title: "Bulls & Cows on FPGA — 숫자야구를 보드 위에"
summary: "논리회로 수업 최종 산출물. 키패드로 세 자리를 넣으면 7세그먼트에 S/B 판정이 뜨고, 피에조가 응원가를 연주합니다. SystemVerilog로 짜서 FPGA 보드에서 실제로 동작."
period: "2023.12"
sortDate: "2023-12-27"
category: "Hardware"
role: "2인 팀"
tags: ["SystemVerilog", "FPGA", "7-segment", "XORShift"]
repo: "https://github.com/kmjstr35/bullsAndCows"
---

## 무엇을 만들었나

세 자리 숫자야구를 순수 하드웨어 로직으로 구현했습니다. 10키 키패드로 서로 다른 숫자 세 개를 넣으면 8자리 7세그먼트에 입력값과 스트라이크·볼 수가 표시되고, 맞히면 "SuccESS", 기회를 다 쓰면 "you LoSE"가 흐릅니다.

- **난수 생성** — 사용자가 먼저 네 자리 시드를 입력하면 그걸로 16비트 XORShift를 돌려 정답을 만듭니다. 시드 입력 상태는 LED로 표시.
- **BGM** — 1 MHz 클럭에서 음계별 분주 값을 테이블로 두고, 피에조로 세 곡(야구장 응원가 등)을 상황에 맞춰 재생합니다.
- 딥스위치로 모드를 바꾸고, RGB LED로 게임 상태를 보여 줍니다.

## 배운 점

blocking/non-blocking 할당을 섞어 쓰면 시뮬레이션과 보드 동작이 달라진다는 걸 몸으로 배웠습니다. 마지막 날 커밋 대부분이 그 정리였습니다.
