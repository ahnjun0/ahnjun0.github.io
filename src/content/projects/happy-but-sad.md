---
title: "기쁜데 슬프다 — AI 감정 셀카 미션 SNS"
summary: "매일 주어지는 목표 감정 조합을 셀피 표정으로 재현하면, 온디바이스 ML이 감정 비율을 분석해 점수를 매기고 그룹 피드에 공유합니다."
period: "2026.06"
sortDate: "2026-06-24"
category: "Web/App"
role: "팀"
tags: ["Kotlin", "Jetpack Compose", "TensorFlow Lite", "ML Kit"]
---

그룹 선택 → 오늘의 감정 미션 확인 → 셀피 촬영 → ML Kit 얼굴 검증(정확히 1명) → TFLite로 HAPPY/SAD/ANGRY/SURPRISED 비율 예측 → 목표 비율과의 차이로 0~100점. 결과 화면에 이미지·감정 비율·점수·피드백·위치 태그를 함께 보여줍니다.
