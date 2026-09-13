---
title: "IRYA — AI mock video-interview service"
summary: "Kakao Tech Campus 4th-cohort team project. WebRTC video interviews with an STT/AI pipeline. I own infrastructure and backend."
period: "Jul 2026 – ongoing"
sortDate: "2026-09-12"
category: "Infra"
role: "Team · infrastructure / backend"
tags: ["LiveKit", "Caddy", "Docker Compose", "WebRTC", "FastAPI"]
repo: "https://github.com/kakaotechcampus-4/ktc4-pusan-1"
---

## Problem

Practice interviews on your own: get questions over video, answer, and receive AI feedback. Real-time audio/video is the core, so the media server and deployment are half the project.

## What I do

- **Self-hosted LiveKit** — a WebRTC media server in containers instead of an external SaaS, merged into the infra directory.
- **Caddy reverse proxy with HTTPS** — automated domains/certificates, per-service routing.
- Docker Compose deployment with dev/prod separation.

This is ongoing; results and lessons will be filled in when it wraps up.
