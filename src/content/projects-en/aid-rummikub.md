---
title: "AID Rummikub — A learned Rummikub agent"
summary: "Beat a greedy ILP baseline in two-player Rummikub (104 tiles). Search is the teacher; the final move choice is the network's."
period: "Jul – Aug 2026"
sortDate: "2026-08-29"
category: "Game AI"
role: "Team (AID)"
result: "Grand Prize"
tags: ["PyTorch", "Distillation", "RL", "ILP"]
---

## Problem

A club competition task. Beat a greedy ILP baseline in two-player Rummikub without jokers, under the constraint that the final agent must be **learned**: a solver may generate legal moves, but the network has to choose among them.

## Approach

- Used the search agent as a **distillation data generator**, not the submission.
- Staged: first a hybrid (network for the midgame, DFS allowed for endgames with ≤ 5 tiles) with an ablation proving the network's contribution, then a pure-network agent.
- RL fine-tuning with sparse reward, critic pre-training, and a rank anchor; a `greedy_hold` baseline and an end-on-stuck option to stabilize the training environment.

## Result

AID Grand Prize.
