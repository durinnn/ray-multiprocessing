"""실험 B: Standalone Ray 구조 + 개선 플래그 (plan.md §3.3).

nested(실험 A)와 동일한 파이프라인을 Triton python backend 밖의 독립 Ray
프로세스에서 실행하고, 개선 항목을 개별 플래그로 on/off 해 기여도를 분해한다.
플래그 전부 off = B0(nested와 동일 로직), 전부 on = B-all.
"""
