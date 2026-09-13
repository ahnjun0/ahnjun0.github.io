---
title: "IRYA — AI 화상면접 지원 서비스"
summary: "카카오테크캠퍼스 4기 팀 프로젝트. WebRTC 화상면접과 STT·AI 파이프라인. 인프라와 백엔드를 맡고 있습니다."
period: "2026.07 – 진행 중"
sortDate: "2026-09-12"
category: "Infra"
role: "팀 · 인프라/백엔드"
tags: ["LiveKit", "Caddy", "Docker Compose", "WebRTC", "FastAPI"]
repo: "https://github.com/kakaotechcampus-4/ktc4-pusan-1"
---

## 문제

면접 연습을 혼자서도 할 수 있게, 화상으로 질문을 받고 답하면 AI가 피드백을 주는 서비스입니다. 실시간 영상·음성이 핵심이라 미디어 서버와 배포가 프로젝트의 절반입니다.

## 맡은 일

- **LiveKit 자체 호스팅** — 외부 SaaS 없이 WebRTC 미디어 서버를 컨테이너로 올리고 인프라 디렉터리로 통합.
- **Caddy 리버스 프록시와 HTTPS** — 도메인·인증서 자동화, 서비스별 라우팅.
- Docker Compose 기반 배포 구성과 개발/운영 분리.

진행 중인 프로젝트라, 마무리되면 결과와 배운 점을 채웁니다.
