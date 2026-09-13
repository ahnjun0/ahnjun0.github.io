---
title: "AID Web — 동아리 웹사이트와 대회 자동채점 시스템"
summary: "부산대 AI 동아리 AID 공식 사이트. 학번 로그인, CSV 제출 자동채점과 리더보드, 마크다운+LaTeX 대회 설명."
period: "2026.04 – 05"
sortDate: "2026-05-05"
category: "Web/App"
role: "회장으로서 기획·구축"
tags: ["Django 5", "PostgreSQL", "Docker", "Nginx", "TailwindCSS"]
repo: "https://github.com/PNU-AID/webpage"
---

## 문제

동아리 소개·세미나·수상 내역을 보여주는 사이트가 필요했고, 동시에 **동아리 내부 대회를 Kaggle처럼 운영**할 수 있는 채점 시스템이 필요했습니다.

## 접근

- 공개 페이지(소개·멤버·프로젝트·세미나·수상·공지)와 회원 시스템(학번 기반 로그인, 초기 비밀번호 강제 변경, 관리자 CSV 일괄 가입).
- 대회 시스템: 대회 생성/관리, CSV 제출 자동 채점(accuracy / f1 / rmse), 리더보드, 마크다운+LaTeX 대회 설명.
- 보안: django-axes 로그인 제한, fail2ban, Nginx rate limiting. 서버 시간대를 KST로 통일.
- Docker Compose + Nginx + Gunicorn으로 배포, django-storages로 S3 전환 가능하게.

## 결과

동아리 운영 기간 동안 실제로 사용했습니다. 이 시스템 위에서 열려던 내부 대회는 내부 사정으로 시행하지 못했습니다.
