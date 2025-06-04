# Advanced Driver Assistance Systems (ADAS) GUI

A comprehensive graphical user interface application implementing multiple Advanced Driver Assistance Systems features using computer vision and deep learning.

![Result](https://user-images.githubusercontent.com/96789016/198597127-f498d315-0767-4ca7-aefe-653c76c93e5c.gif)

## Features

### 1. Traffic Sign Detection
- Real-time detection and classification of 43 different traffic signs
- Uses Convolutional Neural Network (CNN) for accurate sign recognition
- Supports common traffic signs including:
  - Speed limits
  - Stop signs
  - Yield signs
  - Warning signs
  - Mandatory signs
  - Prohibition signs

### 2. Traffic Light Detection
- Real-time traffic light state detection
- Recognition of red, yellow, and green signals
- Visual feedback through the GUI

### 3. Lane Detection
- Real-time lane line detection
- Lane departure warning system
- Visual overlay of detected lanes

## Requirements

```
numpy
pandas
opencv-python (cv2)
cvzone
matplotlib
tensorflow
pillow
scikit-learn
tqdm
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Advanced-Driver-Assistance-Systems-GUI.git
cd Advanced-Driver-Assistance-Systems-GUI
```

2. Install required packages:
```bash
pip install numpy pandas opencv-python cvzone matplotlib tensorflow pillow scikit-learn tqdm
```

## Usage

1. Run the GUI application:
```bash
python GUI.py
```

2. The application window will open with options for:
   - Traffic Sign Detection
   - Traffic Light Detection
   - Lane Detection

3. Use the interface to select the desired ADAS feature and input source (video/camera).


## Model Architecture

The traffic sign detection model uses a CNN architecture with:
- Multiple convolutional layers with ReLU activation
- Max pooling layers
- Batch normalization
- Dropout for regularization
- Dense layers for classification
- 43 output classes for different traffic signs

## Acknowledgments

- Traffic sign dataset: German Traffic Sign Recognition Benchmark (GTSRB)
- OpenCV community for computer vision tools
- TensorFlow team for deep learning framework
