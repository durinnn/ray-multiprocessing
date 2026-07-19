# 실험 A(nested) 본 측정 경로. CPU-only로 실행하며 GPU가 없어도 python backend
# 모델은 정상 동작한다. 베이스 이미지 자체가 수 GB이므로 최초 pull에 시간이 걸린다
# (plan.md §8 — Docker 불가 환경은 benchmark/nested/run_parent.py fallback을 쓴다).
FROM nvcr.io/nvidia/tritonserver:24.08-py3

WORKDIR /workspace

# python backend가 model.py를 읽을 때 ASCII 기본 로케일이면 한글 주석에서
# UnicodeDecodeError로 모델 로드가 실패한다
ENV LC_ALL=C.UTF-8 LANG=C.UTF-8 PYTHONIOENCODING=utf-8

COPY triton/requirements.txt /workspace/triton-requirements.txt
RUN pip install --no-cache-dir -r /workspace/triton-requirements.txt

COPY . /workspace

# triton/models/*/1/model.py가 benchmark.common.* 를 import할 수 있도록
ENV PYTHONPATH=/workspace

EXPOSE 8000 8001 8002

CMD ["tritonserver", "--model-repository=/workspace/triton/models"]
