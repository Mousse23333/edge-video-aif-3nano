FROM robot-project:latest

ENV PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_TRUSTED_HOST=pypi.ngc.nvidia.com
ENV PIP_NO_CACHE_DIR=true
ENV PIP_ROOT_USER_ACTION=ignore

RUN pip3 install ultralytics && pip3 install "numpy==1.26.4"

RUN pip3 install pyyaml

WORKDIR /app
COPY benchmark.py .
COPY benchmark_multi.py .
COPY benchmark_batch.py .
COPY benchmark_dual.py .
COPY benchmark_skip.py .
COPY benchmark_switch.py .
COPY benchmark_lite.py .
COPY benchmark_lite_tuning.py .
COPY diagnose_lite.py .
COPY benchmark_window_skip.py .
COPY benchmark_imgsz.py .
COPY engine/ engine/
COPY controllers/ controllers/
COPY nano_inference_server.py .
COPY config/ config/
COPY run_episode.py .
COPY run_heuristic.py .
COPY run_all_controllers.py .
COPY test_aif_only.py .
COPY run_multi_experiment.py .
COPY run_ablation.py .
COPY run_overnight.sh .

# Pre-download model into /app so it's baked into the image
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

CMD ["python3", "benchmark.py"]
