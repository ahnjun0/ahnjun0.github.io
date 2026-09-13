---
title: "Zero-Matgo — A Matgo AI that learns from the rules alone"
summary: "No human strategy, only win/loss. Determinized MCTS for the hidden opponent hand; a hand-built engine for ppeok, jjok, heundeulgi, and chongtong."
period: "Dec 2025 – Jan 2026"
sortDate: "2026-01-08"
category: "Game AI"
role: "Solo"
tags: ["PyTorch", "MCTS", "Self-play", "AlphaZero"]
featured: 3
---

## Problem

Training an AlphaGo Zero–style agent for Matgo, the Korean flower-card game. Two things differ from Go: **you cannot see the opponent's hand** (imperfect information), and the rules are messy — ppeok, jjok, ttadak, heundeulgi, chongtong, gukjin transformation, pi-bak / gwang-bak / meong-bak, go/stop.

## Approach

- **Pure Zero**: no human heuristics like "take the gwang" or "go three times". The agent values positions from win/loss outcomes only.
- **Determinized MCTS**: each simulation fixes (determinizes) the unseen cards, searches, and the results across determinizations are combined to pick a move.
- **Own rules engine**: standard 48 cards, go/stop at 7+ points, doubling from the third go, heundeulgi/bomb ×2, chongtong instant 10-point win, all the bak penalties — implemented as house rules and pinned with tests.
- Core (rules) / Env (training) / Agent (MCTS + network) are separated so I could swap network architectures (MatgoNet, a temporal variant).

## Result

Self-play training runs, with diagnostic scripts for hand-size vs. win-rate correlation and bias. Quantitative metrics are still being organized, so I'm not listing them here.

## What I learned

- One bug in the rules engine poisons the whole training run. I spent longer testing the engine than training, and that was the right call.
- In imperfect-information games, modeling *what you don't know* is harder than modeling what you do.
