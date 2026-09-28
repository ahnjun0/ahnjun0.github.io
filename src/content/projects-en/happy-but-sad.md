---
title: "Happy but Sad — An AI emotion-selfie mission SNS"
summary: "Recreate the day's target emotion mix with a selfie; on-device ML scores the emotion ratios and shares to a group feed."
period: "Jun 2026"
sortDate: "2026-06-24"
category: "Web/App"
role: "Team · on-device ML inference · network layer"
tags: ["Kotlin", "Jetpack Compose", "TensorFlow Lite", "ML Kit"]
repo: "https://github.com/192cm/gippeunde-seulpeuda"
---

Pick a group → see today's emotion mission → take a selfie → ML Kit checks there's exactly one face → a TFLite model predicts HAPPY/SAD/ANGRY/SURPRISED ratios → score 0–100 from the distance to the target mix. The result screen shows the image, per-emotion ratios, score, feedback, and a location tag.
