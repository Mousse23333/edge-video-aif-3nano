FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_ROOT_USER_ACTION=ignore
RUN apt-get update -qq && \
    apt-get install -y -qq python3-pip libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*
RUN pip3 install "numpy==1.26.4"
RUN pip3 install torch torchvision \
    --index-url http://jetson.webredirect.org/jp6/cu126
RUN pip3 install ultralytics pyyaml opencv-python-headless
WORKDIR /app
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
CMD ["python3", "-c", "import torch; print('CUDA:', torch.cuda.is_available())"]
