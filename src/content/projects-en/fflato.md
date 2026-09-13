---
title: "FFlato — One-click attendance for PLATO"
summary: "No more digging through the PNU LMS attendance pages: enter the code straight from the extension icon."
period: "Sep 2026"
sortDate: "2026-09-11"
category: "Product"
role: "Solo · in use"
result: "In use"
tags: ["Chrome Extension", "Manifest V3", "JavaScript"]
featured: 4
repo: "https://github.com/ahnjun0/fflato"
---

## Problem

Attendance on PNU's PLATO LMS means: course page → attendance menu → find the current session → enter the code. You have about 30 seconds at the start of class to do it, and when the page structure changed, auto-attendance sometimes failed silently.

## Approach

A Manifest V3 Chrome extension. Click the icon and the popup **auto-discovers active attendance sessions** and lets you enter the code right there.

- Deadline countdown based on the server's closing time — it stays correct even if you close and reopen the popup.
- Shows the attempt count the server reports when a code is wrong.
- Handles several courses with open sessions on one screen; sessions you've already attended are excluded.
- A self-diagnostics page shows the scan process and PLATO protocol checks, so when the site changes you can see exactly where it broke.

## Result

In daily use on Chrome, Edge, and Whale (loaded in developer mode).

## What I learned

- For a tool that sits on top of someone else's site, "shows you why it broke" matters more than "works".
