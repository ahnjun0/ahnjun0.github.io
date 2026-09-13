---
title: "cAIendar — Developer events as a calendar feed"
summary: "An operator drops in an announcement URL, Claude extracts the schedule, and only reviewed events are published as an iCal feed (webcal/.ics)."
period: "Jul 2026"
sortDate: "2026-07-08"
category: "Product"
role: "Solo · MVP"
tags: ["FastAPI", "SQLAlchemy", "Claude structured outputs", "icalendar"]
---

## Problem

Hackathon, conference, meetup, and IT-contest announcements are scattered, and moving them into a calendar is manual. I wanted "a calendar you subscribe to that fills itself".

## Approach

1. **Ingest** — announcement URL → SSRF guard → trafilatura body extraction, with a paste-the-text fallback.
2. **Extract** — Claude structured output forces domain fields (event type, deadline type, online/offline), confidence, and supporting quotes. Absolute dates are re-validated with dateutil; contradictions trigger up to two re-prompts and then a review flag.
3. **Review** — an operator console to edit, approve, or reject. Nothing reaches the feed before approval. Original extraction vs. final approval is stored as the source of an edit-rate metric.
4. **Publish** — RFC 5545 iCal. Edits and cancellations propagate to subscribed calendars via SEQUENCE/STATUS.

Tests cover RFC 5545 round-trips, the SSRF guard, date re-validation, and feed/console acceptance criteria.

## Result

Sprint 0–2 MVP complete (data model, ICS feed server, operator console, AI extraction pipeline).
