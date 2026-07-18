# 🧑 3D Face Reconstruction

Upload a frontal and side face image to reconstruct an interactive 3D face mesh in your browser.

## Features

- Upload front + side image pair, or a single image
- Real-time 3D face mesh reconstruction
- Interactive 3D visualization (rotate, zoom)
- Runs entirely on CPU — no GPU required

## How It Works

1. Face detection using OpenCV DNN (ResNet-10 SSD)
2. Face cropping and alignment via similarity transform
3. Position map prediction using a MobileNet-v2 based network (ONNX)
4. 3D mesh extraction and colorization from the input image

## Local Development

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Acknowledgement

Originally cloned from [olalium/face-reconstruction](https://github.com/olalium/face-reconstruction) and modified with different face detectors and  deployed in streamlit.
