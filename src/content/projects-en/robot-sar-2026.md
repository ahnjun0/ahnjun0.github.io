---
title: "Search & Rescue AMR — finding runaway apples without GNSS"
summary: "Maps an apartment from wheel encoders, LiDAR and a camera alone, finds both red apples, and returns to the start within 8.9 cm."
period: "Sep 2026"
sortDate: "2026-09-30"
category: "AI/ML"
role: "Team of 3 ('Not a Robot') · all driving software"
result: "3rd Prize · Dept. of CSE Head's Award"
tags: ["Webots", "SLAM", "Scan matching", "D*", "DWA", "YOLO"]
featured: 2
---

## Problem

2026 CSE TECH WEEK Search & Rescue hackathon for autonomous mobile robots. In a Webots R2025a apartment world, a TurtleBot3 Burger has to **leave the start point, find and approach two red apples, and come back to where it started**.

The constraints were the point: **no map is given, and GNSS or any absolute position is banned**, with a cap on the robot's default speed. That leaves wheel encoders, a 2D LiDAR, a camera, and a compass and gyro. A pedestrian walks the apartment at 0.2 m/s, contact costs points, and the judging criteria explicitly included keeping "a sufficient safety margin".

## Approach

Driving is a `SCAN → EXPLORE ⇄ APPROACH → RETURN → DONE` state machine, and each stage closes off the one failure that hurt most.

- **Localization** — encoders alone accumulate drift. LiDAR scan matching corrects x and y continuously, and the wheel radius is calibrated to a measured 0.0336 m. Comparing wheel rotation against the compass **detects slip**, and the distance travelled during a slip is not trusted.
- **Exploration** — pick an unexplored frontier, plan with D*, but issue velocity commands through DWA.
- **Pedestrian** — detect the person with the camera, track them with a Kalman filter, and route around their predicted path.
- **Low obstacles** — floor objects (fruit, cans) that the LiDAR cannot see are picked up by the camera and YOLO and written into the obstacle map.
- **Target confirmation** — colour segmentation proposes candidates, then YOLO confirms each one is an apple.

## Results

Measured under the same lighting as the competition, with the final settings.

Table: Results with the final settings

| Metric | Result |
|---|---|
| Red apples | **2 / 2 visited** (in two of three runs of the same settings) |
| Max localization error | **8.9 cm** (304 cm with scan matching and slip detection off) |
| Return | 609.1 s, **0.22 m** from the true start point |
| Pedestrian contact | 0 s |
| Floor objects displaced | 0 / 8 (3 / 8 with camera obstacles off) |

Measurement is done by a separate script that reads ground truth through a Supervisor, and **never feeds the driving logic**. The submission flattens the modules into a single file, verified to produce exactly the same state log as the module run. If any trace of GPS or Supervisor reaches the submission, the build itself fails.

## What I learned

- Running the same world with a feature on and off, and **attributing the gain separately**, was the only way to settle the configuration. I kept the features where the numbers actually split, like 304 cm → 8.9 cm, and turned the rest off.
- Even with a deterministic simulator, **two out of three runs** has to be reported as two out of three. I did not quote the one good run as the result.
- The remaining weaknesses are measured too. A 2 cm carpet edge is invisible to the LiDAR, so it is detected but not avoided in advance, and in wide corridors DWA alternates left and right. Diagnostics showed 15 of 19 such stretches had a fully safe forward candidate, so it is a direction-choice problem, not a clearance one.
