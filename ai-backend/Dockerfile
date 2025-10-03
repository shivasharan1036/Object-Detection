FROM python:3.9-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

RUN mkdir -p temp_uploads outputs yolo-config

# Download YOLO files
WORKDIR /app/yolo-config
RUN wget -q https://pjreddie.com/media/files/yolov3.weights && \
    wget -q https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg && \
    wget -q https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names

WORKDIR /app

EXPOSE 5002

CMD ["python", "app.py"]