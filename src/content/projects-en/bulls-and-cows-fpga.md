---
title: "Bulls & Cows on FPGA — the number game, in hardware"
summary: "Final deliverable for a logic design course. Enter three digits on a keypad, read strikes and balls on 7-segment displays, and a piezo plays a ballpark chant. Written in SystemVerilog and running on a real FPGA board."
period: "Dec 2023"
sortDate: "2023-12-27"
category: "Hardware"
role: "Team of 2"
tags: ["SystemVerilog", "FPGA", "7-segment", "XORShift"]
repo: "https://github.com/kmjstr35/bullsAndCows"
---

## What it is

A three-digit Bulls & Cows game built entirely in hardware logic. Three distinct digits go in through a 10-key pad; an 8-digit 7-segment display shows the input and the strike/ball count, then scrolls "SuccESS" or "you LoSE".

- **Random numbers** — the player first enters a four-digit seed, which drives a 16-bit XORShift generator to produce the answer. Seed-entry progress shows on LEDs.
- **BGM** — note frequencies are stored as clock-divider values at 1 MHz, and a piezo plays one of three tunes (including a ballpark chant) depending on game state.
- DIP switches switch modes; RGB LEDs show game state.

## What I learned

Mixing blocking and non-blocking assignments makes simulation and the board disagree. Most of the last-day commits were cleaning exactly that up.
