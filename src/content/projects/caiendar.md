---
title: "cAIendar — 개발자 행사 공고를 캘린더 피드로"
summary: "운영자가 공고 URL을 넣으면 Claude가 일정을 추출하고, 검수·승인된 행사만 iCal 피드(webcal/.ics)로 발행합니다."
period: "2026.07"
sortDate: "2026-07-08"
category: "Product"
role: "개인 · MVP"
tags: ["FastAPI", "SQLAlchemy", "Claude structured outputs", "icalendar"]
---

## 문제

해커톤·컨퍼런스·밋업·IT 공모전 공고는 흩어져 있고, 캘린더에 옮기는 건 손으로 합니다. "구독하면 자동으로 들어오는 캘린더"를 만들고 싶었습니다.

## 접근

1. **수집** — 공고 URL 투입 → SSRF 가드 → trafilatura 본문 추출. 실패 시 본문 붙여넣기 폴백.
2. **추출** — Claude structured output으로 행사 유형·마감 유형·온오프라인 등 도메인 필드와 confidence, 근거 인용을 강제. 절대 날짜는 dateutil로 재검증하고 모순 시 최대 2회 재프롬프트 후 검수 플래그.
3. **검수** — 운영자 콘솔에서 수정·승인·반려. 승인 전에는 어떤 결과도 피드에 노출되지 않음. 원본 추출 vs 최종 승인을 저장해 수정률 지표의 원천으로.
4. **발행** — RFC 5545 iCal. 수정·취소는 SEQUENCE/STATUS로 구독 캘린더에 전파.

테스트: RFC 5545 왕복, SSRF 가드, 날짜 재검증, 피드·콘솔 인수 기준.

## 결과

스프린트 0–2 MVP 완성(데이터 모델, ICS 피드 서버, 운영자 콘솔, AI 추출 파이프라인).
