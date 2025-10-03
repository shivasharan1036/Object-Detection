# Structure of the microservice
object-detection-microservice/
   ui-backend/
        app.py
        requirements.txt
        Dockerfile
   ai-backend/
        app.py
        requirements.txt
        Dockerfile
        yolo-config/
  docker-compose.yml
        README.md


# Object Detection Microservice

A microservice architecture for object detection using YOLOv3, consisting of:
- **UI Backend**: Handles image uploads and user interactions
- **AI Backend**: Performs object detection using YOLOv3 model
