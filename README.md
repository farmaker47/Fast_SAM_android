# FastSAM Image Segmentation Android Demo

### Overview

This is a camera app that continuously segment the objects in the frames seen
by your device's camera.

# Model

[FastSAM by Ultralytics](https://docs.ultralytics.com/models/fast-sam/) is a real-time, CNN-based model designed to segment any object in an image with minimal computational resources. It builds on YOLOv8-seg and is tailored for high-speed and efficient segmentation across various tasks.

## Key Features
- Real-time segmentation using CNNs
- Efficient instance segmentation via prompt-guided selection (not applicable in this android demo)
- Built on YOLOv8-seg for fast and accurate performance

## Installation
```bash
git clone https://github.com/CASIA-IVA-Lab/FastSAM.git
cd FastSAM
pip install -r requirements.txt



