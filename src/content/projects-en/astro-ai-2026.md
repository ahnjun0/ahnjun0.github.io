---
title: "3rd Astronomy & Space AI Competition — 72-hour solar wind forecasting"
summary: "Predict solar wind speed 6–72 hours ahead from five days of solar imagery (193 Å / 211 Å) and past wind speed."
period: "Aug 2026"
sortDate: "2026-08-21"
category: "AI/ML"
role: "Team 'Aing-Aing' · shared across the stack"
result: "5th place"
tags: ["Time series", "Imagery", "PyTorch", "RMSE"]
link: "https://kaist-kasiai.elice.io/"
---

## Problem

Run by the Korea Astronomy and Space Science Institute and the KAIST SW Education Center as part of the SpaceAI program. Each sample gives 20 time steps of two solar image bands and 20 wind-speed readings; the target is wind speed at 12 six-hourly steps. The metric is RMSE, which punishes large errors — miss a high-speed stream and you drop several places.

## Approach

Train 9,607 / Val 1,199 / Test 3,868 samples. Validation was for model selection only and retraining on Train+Val was prohibited, so we fixed the validation strategy first and then experimented with models combining image features and the time series.

## Result

**5th place** among the 30 finalist teams (Aug 18–21), one of five awarded teams.

The team repository lives in the [Aing-Aing](https://github.com/Aing-Aing) organization (private).
