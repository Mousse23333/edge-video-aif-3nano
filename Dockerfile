FROM robot-project:latest

ENV PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_TRUSTED_HOST=pypi.ngc.nvidia.com
ENV PIP_NO_CACHE_DIR=true
ENV PIP_ROOT_USER_ACTION=ignore

RUN pip3 install ultralytics && pip3 install "numpy==1.26.4"

RUN pip3 install pyyaml

WORKDIR /app
COPY engine/ engine/
COPY controllers/ controllers/
COPY config/ config/
COPY nano_inference_server.py .
COPY test_aif_only.py .
COPY run_multi_experiment.py .
COPY run_ablation.py .
COPY run_overnight.sh .

# Pre-download model into /app so it's baked into the image
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

CMD ["bash", "run_overnight.sh"]
