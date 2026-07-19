# Ray Nested vs Standalone 벤치마크 레포

**목표**: Triton Python Backend 내부에 Ray를 nested로 기동한 CCTV 분석 파이프라인에서 관찰된 성능 문제를 회사 코드 없이 합성 워크로드로 재현하고, Standalone Ray + 직접 gRPC 재설계의 개선 효과를 수치로 검증합니다.

## 개요

이 레포는 세 가지 산출물을 포함합니다:

1. **두 구조(nested / standalone)를 동일 워크로드로 실행할 수 있는 실험 코드** (`benchmark/nested`, `benchmark/standalone`)
2. **Prometheus + Grafana 기반 측정 환경과 재현 스크립트** (`monitoring/`, `scripts/`)
3. **문제정의 → 가설 → 측정 → 결과 → 한계 구조의 문서** (`docs/`, `plan.md`, `research.md`, `proposal.md`)

## 빠른 시작

(Phase 1~3 구현 완료, Phase 4~6 진행 중 — 최종 결과·본문은 Phase 6에서 갱신)

```bash
bash scripts/smoke.sh              # 뼈대 동작 확인
bash scripts/start_monitoring.sh   # 모니터링 스택 기동 (선택)
bash scripts/run_nested.sh         # 실험 A (Triton 경로, 최초 실행 시 이미지 pull에 수 GB 소요)
bash scripts/run_nested.sh --fallback  # 실험 A (Docker 불가 환경용 parent-process fallback)
```

## 구조

```
├── plan.md                    # 이 벤치마크의 실행 계획
├── research.md                # 원 시스템의 12개 문제 분석
├── proposal.md                # 재설계 타당성 검토
├── config/
│   └── default.yaml           # 모든 실험 설정 중앙화
├── benchmark/
│   ├── common/                # 공유 모듈 (config, frame 생성기, stages, mock_latency, metrics)
│   ├── nested/                # 실험 A: nested Ray Actor 정의 + Docker 불가용 fallback
│   ├── standalone/            # 실험 B: standalone Ray 구조
│   └── micro/                 # 마이크로벤치
├── triton/                    # 실험 A 본 측정 경로: tritonserver 모델 저장소 + Dockerfile
├── inference_mock/            # gRPC 모의 추론 서버 (fallback/실험 B 경로)
├── monitoring/                # Prometheus + Grafana
├── scripts/                   # 실행 스크립트
├── analysis/                  # 결과 분석 (CSV → 그래프)
└── docs/                      # 01-problem, 02-hypotheses, 03-results
```

## 요구사항

- Python 3.11+
- Docker & Docker Compose (monitoring 및 실험 A 본 측정 경로용. 불가 시 `--fallback` 사용)
- 6 CPU cores / 8GB RAM 이상 (로컬 테스트 기준)
- 실험 A 본 측정 경로는 `nvcr.io/nvidia/tritonserver` 이미지(수 GB)를 최초 1회 내려받는다

## 라이선스

Internal use only.
