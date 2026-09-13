---
title: "AID Web — Club website with a competition auto-grader"
summary: "Official site of AID, the PNU AI club. Student-ID login, CSV submission auto-grading with a leaderboard, Markdown + LaTeX contest pages."
period: "Apr – May 2026"
sortDate: "2026-05-05"
category: "Web/App"
role: "Planned and built as president"
tags: ["Django 5", "PostgreSQL", "Docker", "Nginx", "TailwindCSS"]
repo: "https://github.com/PNU-AID/webpage"
---

## Problem

The club needed a site for its intro, seminars, and awards — and at the same time a grading system to **run internal competitions Kaggle-style**.

## Approach

- Public pages (about, members, projects, seminars, awards, notices) and a member system (student-ID login, forced initial password change, admin bulk-signup via CSV).
- Competition system: create/manage contests, auto-grade CSV submissions (accuracy / F1 / RMSE), leaderboard, Markdown + LaTeX descriptions.
- Security: django-axes login throttling, fail2ban, Nginx rate limiting. Server timezone unified to KST.
- Deployed with Docker Compose + Nginx + Gunicorn; django-storages so it can move to S3.

## Result

Used in production while I ran the club. The internal competition planned to run on it was cancelled for internal reasons.
