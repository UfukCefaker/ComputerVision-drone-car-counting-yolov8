# ComputerVision-drone-car-counting-yolov8

# Drone-Based Car Detection and Counting using YOLOv8

This repository contains the implementation of **car detection and counting** from drone-based parking lot images using **YOLOv8**. The project was completed as part of the BBM418 - Computer Vision Laboratory course at Hacettepe University.

## 📌 Objective

The goal of this assignment is to:
- Detect car objects from aerial images captured by drones
- Count the number of cars in each image
- Evaluate object detection performance using metrics such as exact match accuracy and mean squared error (MSE)
- Compare the effect of different training strategies and hyperparameters using the YOLOv8 object detection model

## ⚙️ Approach

- The task is performed using **YOLOv8n**, a lightweight version optimized for speed and edge deployment.
- Four training configurations are tested by freezing different portions of the model (from partial freezing to full fine-tuning).
- Various hyperparameters (learning rate, optimizer, batch size, etc.) are explored.
- Final predictions are compared against ground truth annotations for analysis and evaluation.


## 🔒 Academic Disclaimer

This work is part of a university course assignment and follows strict academic integrity guidelines. No part of the implementation should be reused without proper attribution.

---

> Created by **Ufuk Cefaker**  
> Hacettepe University, Computer Engineering  
