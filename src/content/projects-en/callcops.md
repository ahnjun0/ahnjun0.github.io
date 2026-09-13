---
title: "CallCops — Real-time audio watermarking for the telephone network"
summary: "Embeds an inaudible 128-bit watermark into 8 kHz call audio and recovers it with under 5% bit error rate after G.729 compression."
period: "Jan – Feb 2026"
sortDate: "2026-02-01"
category: "AI/ML"
role: "Team (KAIST Madcamp) · model design & training"
tags: ["PyTorch", "Causal Conv", "Codec simulator", "ONNX Runtime"]
repo: "https://github.com/ahnjun0/callcops"
---

## Problem

A system that inserts an **inaudible digital signature** into call audio in real time and detects it on the other end, for voice-phishing prevention and call-integrity verification. The telephone network is narrowband (8 kHz) and runs through aggressive codecs like G.711/G.729, so ordinary audio watermarking doesn't survive.

## Approach

- Encoder / Decoder / Discriminator. Causal convolutions keep latency under 200 ms.
- A **differentiable codec simulator inside the training loop**, so the watermark learns to survive compression. Curriculum learning ramps the difficulty.
- A composite loss balancing audio quality (PESQ ≥ 4.0) against detection rate.
- A lightweight model (< 10 MB) exported for ONNX Runtime Web/Mobile so it runs on-device without a server.

## Result

Target spec: latency < 200 ms, PESQ ≥ 4.0, BER < 5% after G.711/G.729. Built as one week's project of the four-week Madcamp, with a prototype and a preview web app (`callcops-preview`).

## What I learned

- You can't "evaluate after the codec" — the codec has to be *inside* training for the watermark to survive it.
