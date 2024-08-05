import cv2 as cv
from ultralytics import YOLO
import os

dataset = 'train/data.yaml'

# load a model
# backbone = YOLO("yolov8n.yaml")  # build a new model from scratch
backbone = YOLO("yolov8n.pt")  # load a pre-trained model (recommended for training)

# Use the model
results = backbone.train(data=dataset, epochs=20)  # train the model