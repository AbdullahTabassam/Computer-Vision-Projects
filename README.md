# Computer Vision Projects Repository

Welcome to my Computer Vision Projects Repository. This repository contains a collection of projects that I have worked on in the field of computer vision. Each project demonstrates my skills and experience in applying computer vision techniques to solve real-world problems. 

## Contents

1. [In-flight Bird Species Detection and Counting](https://github.com/AbdullahTabassam/In-flight_Bird_Species_Detection_Counting_)
2. [3D Reconstruction Project](https://github.com/AbdullahTabassam/3D-Recontruction)
3. [Lung Tumor Detection using YOLOv8 Segmentation Models and Flask Web Application](https://github.com/AbdullahTabassam/Lungs_Tumor_Segmentation_Computer_Vision)
4. [Object Detection Model for Bird Species Identification](https://github.com/AbdullahTabassam/Custom-Object-Detection)
5. [UK Garden Birds Detection using YOLO v8 model](https://github.com/AbdullahTabassam/Yolo_Bird_detection/)
6. [Hand Gesture Volume Control](https://github.com/AbdullahTabassam/Volume-Control-Using-Computer-Vision)
7. [Object Detection Model for Sign Language Identification](https://github.com/AbdullahTabassam/Sign_language_Object_Detection)
8. [ORB Feature Mapping](https://github.com/AbdullahTabassam/Feature_Mapping_ORB_Computer_Vision)
9. [Eulerian Magnification for Heart Rate Measurement](https://github.com/AbdullahTabassam/Feature_Mapping_ORB_Computer_Vision)

---

### Project 1: In-flight Bird Species Detection and Counting

Welcome to the repository for the In-flight Bird Species Detection and Counting project! This project, developed as part of an MSc thesis, focuses on utilizing object detection models for the categorization, species detection, and counting of birds. By leveraging computer vision techniques, this project aims to contribute to wildlife conservation and research efforts.

#### Demo
![Demo_gif](https://github.com/AbdullahTabassam/In-flight_Bird_Species_Detection_Counting_/blob/master/images/Birdtrackingandcounting.gif)
##### Images
A few detection images are as follows:
<div>
<img src="https://github.com/AbdullahTabassam/In-flight_Bird_Species_Detection_Counting_/blob/master/images/Image%20(1).jpg" alt="Image" height="250" width="250">
<img src="https://github.com/AbdullahTabassam/In-flight_Bird_Species_Detection_Counting_/blob/master/images/Image%20(2).jpg" alt="Image" height="250" width="250">
<img src="https://github.com/AbdullahTabassam/In-flight_Bird_Species_Detection_Counting_/blob/master/images/Image%20(3).jpg" alt="Image" height="250" width="250">
</div>
<div>
<img src="https://github.com/AbdullahTabassam/In-flight_Bird_Species_Detection_Counting_/blob/master/images/Image%20(5).jpg" alt="Image" height="250" width="250">
<img src="https://github.com/AbdullahTabassam/In-flight_Bird_Species_Detection_Counting_/blob/master/images/Image%20(10).jpg" alt="Image" height="250" width="250">
<img src="https://github.com/AbdullahTabassam/In-flight_Bird_Species_Detection_Counting_/blob/master/images/Image%20(4).jpg" alt="Image" height="250" width="250">
</div>

#### Overview

This project explores the use of object detection models to identify and categorize various bird species from photos. It includes two primary models: a specific bird species classifier capable of detecting six different bird species and a generic bird classifier categorizing all birds under one class - Birds. The project encompasses data preprocessing, model training, evaluation, and inference.

#### Repository Contents

- **Thesis**: Find the detailed thesis document in the 'Thesis' folder.
- **Notebooks**: All Jupyter notebooks used in the project are available.
- **Code**: All code related to data preprocessing, model training, and evaluation, including the inference script 'inf.py', is available in the repository.
- **Models**: Trained models used in the project are present in the repository.
- **Images**: View detections on images in the 'images' folder.

#### Key Features

- Utilizes fine-tuning of YOLOv8 model for accurate bird detection.
- Implements transfer learning for efficient model training.
- Evaluates models using metrics such as precision, recall, and F1-score.
- Provides visualizations including confusion matrices and precision-recall curves for better understanding of model performance.

#### Practical Implications

The object-detection models developed in this project have significant practical implications for conservation and wildlife monitoring activities. By automating bird species identification, researchers can collect data at scale, contributing to a better understanding of bird populations and migration patterns.

**Disclaimer**: This project is developed for educational and research purposes. 

*This project is developed by Abdullah Tabassam as part of an MSc thesis project. For inquiries, please contact the project owner via LinkedIn.*

[![Go to In-flight Bird Species Detection and Counting Repository](https://img.shields.io/badge/Repository-Link-blue)](https://github.com/AbdullahTabassam/In-flight_Bird_Species_Detection_Counting_)

---

### Project 2: 3D Reconstruction Project

#### Overview

This project focuses on utilizing computer vision techniques to reconstruct 3D scenes from monocular images captured by a single camera. The pipeline involves camera calibration, depth map generation, creation of RGBD images, point cloud generation, and trajectory estimation for multiple images. The ultimate goal is to combine these components to produce a complete 3D representation of the scene.

#### Color Image, Depth Estimation, and 3D Point Cloud

<div>
<img src="https://github.com/AbdullahTabassam/3D-Recontruction/blob/main/model_images/images/new.jpg" alt="Image"  width="250">
<img src="https://github.com/AbdullahTabassam/3D-Recontruction/blob/main/model_images/depth/new.png" alt="Image"  width="250">
<img src="https://github.com/AbdullahTabassam/3D-Recontruction/blob/main/3D-PointCloud.png" alt="Image"  width="250">
</div>

#### Features

- **Camera Calibration**: The camera intrinsic and extrinsic parameters are calibrated to accurately interpret the geometry of the scene.
- **Depth Map Generation**: Utilizing Transformer-based vision models, depth maps are generated from monocular images. Initially, Hugging Face Spaces platform is used, with future integration planned for MIDAS v3.1 model to enhance depth map quality.
- **RGBD Image Creation**: Depth maps are combined with corresponding RGB images to form RGBD images, providing both color and depth information.
  
    <img src="https://github.com/AbdullahTabassam/3D-Recontruction/blob/main/DepthMap.png" alt="Image"  width="600">
  
- **Point Cloud Generation**: RGBD images are processed to extract point clouds, representing the scene in a 3D coordinate system.
- **Trajectory Estimation**: ORB features and RANSAC will be employed to map multiple images and estimate rotational and translational parameters for each image. This information will be used to create a trajectory file, facilitating the alignment of multiple point clouds to reconstruct the complete scene in 3D.

#### Libraries Required
- Open3D
- OpenCV-Python
- Numpy
- Matplotlib
- Scikit-Image

#### Skills Demonstrated

- Proficiency in camera calibration techniques.
- Experience with Transformer-based vision models for depth map generation.
- Familiarity with RGBD image processing.
- Understanding of point cloud generation from RGBD images.
- Knowledge of feature matching algorithms for trajectory estimation.
- Ability to integrate multiple components into a cohesive pipeline for 3D reconstruction.

#### Future Work

- Integration of MIDAS v3.1 model for improved depth map generation.
- Optimization of trajectory estimation algorithms for better accuracy and efficiency.
- Exploration of additional techniques for scene reconstruction, such as bundle adjustment.



[![Go to 3D Reconstruction Repository](https://img.shields.io/badge/Repository-Link-blue)](https://github.com/AbdullahTabassam/3D-Recontruction)

---

### Project 3: Lung Tumor Detection using YOLOv8 Segmentation Models and Flask Web Application

This project aims to develop an end-to-end solution for lung tumor detection using YOLOv8 segmentation models and a Flask web application. The model is trained on CT scan images collected from various sources to accurately identify tumors within lung scans. The Flask web application enhances usability and reduces processing time, providing a seamless experience for users.

#### Key Features

- Utilizes YOLOv8 segmentation models for accurate tumor detection in lung CT scan images.
- Hyperparameter tuning performed to optimize model performance, ensuring superior results.
- Flask web application created to provide an intuitive interface for users.
- End-to-end solution allows for efficient tumor detection without the need for complex setups.

#### Demo

##### Web Application

![Demo_gif](https://github.com/AbdullahTabassam/Lungs_Tumor_Segmentation/blob/master/Demo.gif)

#### Results

<div>
<img src="https://github.com/AbdullahTabassam/Lungs_Tumor_Segmentation/blob/master/static/images/000117.png" alt="Image" height="250" width="250">
<img src="https://github.com/AbdullahTabassam/Lungs_Tumor_Segmentation/blob/master/static/images/000157.png" alt="Image" height="250" width="250">
<img src="https://github.com/AbdullahTabassam/Lungs_Tumor_Segmentation/blob/master/static/images/000163.png" alt="Image" height="250" width="250">
</div>
<div>
<img src="https://github.com/AbdullahTabassam/Lungs_Tumor_Segmentation/blob/master/static/results/predicted_000117.png" alt="Image" height="250" width="250">
<img src="https://github.com/AbdullahTabassam/Lungs_Tumor_Segmentation/blob/master/static/results/predicted_000157.png" alt="Image" height="250" width="250">
<img src="https://github.com/AbdullahTabassam/Lungs_Tumor_Segmentation/blob/master/static/results/predicted_000163.png" alt="Image" height="250" width="250">
</div>


#### Libraries Required

- Flask
- YOLOv8
- OpenCV
- Numpy

#### Skills Demonstrated

- Proficiency in using YOLOv8 for segmentation tasks.
- Experience with Flask for developing web applications.
- Ability to integrate machine learning models into web applications.
- Knowledge of hyperparameter tuning for model optimization.



[![Go to Lung Tumor Detection Repository](https://img.shields.io/badge/Repository-Link-blue)](https://github.com/AbdullahTabassam/Lung-Tumor-Detection-using-Yolov8-Segmentation-Models-and-Flask-Web-Application-)

---

### Project 4: Object Detection Model for Bird Species Identification

#### Overview

This project involves the development and deployment of an object detection model for identifying various bird species. The goal is to create a model that can accurately detect and classify bird species from images, aiding in wildlife monitoring and conservation efforts.

#### Key Features

- Utilizes YOLOv8 model for object detection.
- Fine-tuned on a dataset of bird species images for improved accuracy.
- Includes data preprocessing, model training, and evaluation scripts.
- Provides visualizations of detection results on test images.

#### Demo

##### Images
![Birds](https://github.com/AbdullahTabassam/Custom-Object-Detection/blob/master/Screenshots/Detections.jpg)


#### Libraries Required

- YOLOv8
- OpenCV
- Numpy

#### Skills Demonstrated

- Proficiency in object detection using YOLOv8.
- Experience in fine-tuning models for specific tasks.
- Knowledge of data preprocessing techniques for object detection.
- Ability to evaluate model performance using appropriate metrics.


[![Go to Object Detection Repository](https://img.shields.io/badge/Repository-Link-blue)](https://github.com/AbdullahTabassam/Object-Detection-Model-for-Bird-Species-Identification)

---

### Project 5: UK Garden Birds Detection using YOLO v8 model


### Overview

This project is a continuation of a previous endeavor that utilized the TensorFlow Object Detection API and employed SSD and Faster R-CNN models to detect four common bird species found in the UK. In this iteration, the latest YOLO v8 model is used to significantly enhance the detection performance, achieving higher FPS (frames per second) and mAP (mean average precision) values compared to the previous models.

### Key Features

- **Advanced Model**: Utilizes the latest YOLO v8 model for superior object detection performance.
- **Enhanced Speed**: Achieves a higher FPS, making real-time detection feasible.
- **Improved Accuracy**: Provides better mAP values, indicating more precise detections.

### Demo

![Demo_gif](https://github.com/AbdullahTabassam/Yolo_Bird_detection/blob/master/Detections_Birds.gif)


### Features

- **Bird Species**: Detects four common UK bird species with high accuracy.
- **Real-time Detection**: Optimized for real-time applications with improved processing speed.
- **Comparative Analysis**: Demonstrates significant performance improvements over SSD and Faster R-CNN models.

### Comparative Performance

- **FPS**: Achieved a higher FPS, making the YOLO v8 model suitable for applications requiring real-time detection.
- **mAP**: Increased mAP value, indicating more precise and reliable detections.

### Libraries Required

- YOLO v8
- OpenCV
- Numpy

### Skills Demonstrated

- Proficiency in using advanced object detection models.
- Experience in optimizing models for real-time applications.
- Ability to perform comparative analysis of different models to demonstrate improvements.
- Knowledge of data preprocessing and evaluation techniques for object detection.

[![Go to YOLO Object Detection Repository](https://img.shields.io/badge/Repository-Link-blue)](https://github.com/AbdullahTabassam/Yolo_Bird_detection)
 

### Project 6: Hand Gesture Volume Control

#### Introduction

This project implements a Python script for controlling computer volume using hand gestures detected via a webcam. It utilizes MediaPipe and OpenCV libraries for hand tracking and PyCaw library for volume control, specifically designed for Windows operating systems.

#### Demo

![Hand Gesture Volume Control](https://github.com/AbdullahTabassam/Volume-Control-Using-Computer-Vision/blob/master/handGesture.gif)

#### Libraries Used

- MediaPipe
- OpenCV (cv2)
- PyCaw
- Python 3.x

#### Features

- **Real-time Hand Gesture Detection**: Tracks hand gestures using MediaPipe library from webcam feed.
- **Volume Control**: Adjusts system volume based on detected hand gestures (thumb and index finger open/close).
- **Cross-platform Compatibility**: Designed to work specifically on Windows OS due to PyCaw library dependency.
- **User Interface**: Displays live webcam feed with overlaid hand gesture recognition results.
- **Customization**: Allows customization of webcam index and other parameters for different setups.

#### Skills Demonstrated

- **Computer Vision**: Application of MediaPipe and OpenCV for real-time hand gesture recognition.
- **Library Integration**: Integration of PyCaw for controlling system volume via Python script.
- **User Interface Development**: Simple GUI development for visualizing hand gesture detection results.
- **System Interaction**: Interaction with system-level controls (volume adjustment) via Python.
- **Cross-platform Development**: Adaptation and limitations in Windows-specific libraries and functionality.

[![Go to Repository](https://img.shields.io/badge/Repository-Link-blue)](https://github.com/AbdullahTabassam/Volume-Control-Using-Computer-Vision)


### Project 7: Object Detection Model for Sign Language Identification

#### Introduction

This project focuses on training a custom object detection model using the TensorFlow 2 Object Detection API to identify all 26 English alphabets in sign language gestures. The model is based on the Faster R-CNN algorithm with a ResNet101 backbone, enabling real-time detection and recognition of sign language gestures.

#### Demo

![Sign Language Detection](https://github.com/AbdullahTabassam/Sign_language_Object_Detection/blob/master/Screenshots/Detections.jpg)

#### Libraries Used

- TensorFlow 2
- TensorFlow Object Detection API
- OpenCV
- NumPy

#### Features

- **Custom Object Detection**: Trains a model to detect 26 different sign language gestures corresponding to English alphabets.
- **TensorFlow 2 Object Detection API**: Utilizes TensorFlow 2 for efficient model training and inference.
- **Dataset Handling**: Processes datasets in COCO format, sourced from Roboflow, for training and evaluation.
- **Performance Evaluation**: Computes metrics such as mAP (mean Average Precision) to assess detection accuracy.
- **Inference**: Implements inference on unseen images to demonstrate real-world applicability.

#### Skills Demonstrated

- **TensorFlow 2**: Proficiency in using TensorFlow 2 for deep learning model development.
- **Object Detection Algorithms**: Understanding and implementation of Faster R-CNN for object detection tasks.
- **Dataset Handling**: Processing and utilizing datasets in COCO format for training custom models.
- **Model Evaluation**: Metrics computation and evaluation techniques for object detection models.
- **Transfer Learning**: Leveraging pre-trained models from TensorFlow 2 Model Zoo for efficient model training.


[![Go to Repository](https://img.shields.io/badge/Repository-Link-blue)](https://github.com/AbdullahTabassam/Sign_language_Object_Detection)



### Project 8: ORB Feature Mapping Project

#### Introduction

This project demonstrates proficiency in computer vision techniques, specifically feature mapping, using the OpenCV library in Python. It employs the ORB (Oriented FAST and Rotated BRIEF) algorithm to detect and match distinctive features across images or video frames, suitable for applications like object detection and image retrieval.

#### Demo


<img src="https://miro.medium.com/v2/resize:fit:1358/0*5tH4g-DWevzcs_8Y.jpg" alt="Image">

#### Libraries Used

- OpenCV (cv2)
- NumPy
- Operating System (os) module

#### Features

- **Feature Detection**: Extracts key features using the ORB algorithm.
- **Feature Matching**: Matches features across different images or frames.
- **Real-time Object Identification**: Detects and labels objects in real-time webcam video.
- **Modularity**: Easily extendable for additional features or algorithms.
- **Robustness**: Ensures reliable feature detection across varying conditions.

#### Skills Demonstrated

- **OpenCV**: Proficiency in using OpenCV for computer vision tasks, specifically feature detection and matching.
- **Algorithm Implementation**: Implementation of the ORB algorithm for feature extraction and matching.
- **Real-time Applications**: Development of applications for real-time object detection and identification.
- **Modular Programming**: Modular design for easy integration and extension of computer vision algorithms.
  
[![Go to Repository](https://img.shields.io/badge/Repository-Link-blue)](https://github.com/AbdullahTabassam/Feature_Mapping_ORB_Computer_Vision)


### Project 9: Eulerian Magnification for Heart Rate Measurement

#### Introduction

This project implements Eulerian Magnification to amplify color variations in video frames, focusing on facial regions to measure heart rate non-invasively. Leveraging computer vision techniques and OpenCV, it detects subtle changes in skin color caused by blood flow for heart rate estimation.

#### Demo

No specific demo image provided; project focuses on algorithmic implementation and analysis.

#### Libraries Used

- OpenCV
- NumPy
- MediaPipe
- CVZone (additional package)

#### Features

- **Face Recognition**: Utilizes MediaPipe for facial region extraction.
- **Frequency-based Filtering**: Applies Gaussian and Laplacian pyramids for color variation analysis.
- **Heart Rate Estimation**: Estimates heart rate from frequency analysis of facial color changes.
- **Non-invasive Measurement**: Provides non-intrusive method for heart rate measurement using video data.

#### Skills Demonstrated

- **Computer Vision Techniques**: Application of Eulerian Magnification for subtle color variation amplification.
- **Algorithm Implementation**: Implementation of frequency-based filtering using Gaussian and Laplacian pyramids.
- **Biomedical Applications**: Application of computer vision in biomedical fields for heart rate estimation.
- **Package Integration**: Integration and utilization of MediaPipe and CVZone for facial recognition and analysis.

[![Go to Eulerian Magnification for Heart Rate Measurement Repository](https://img.shields.io/badge/Repository-Link-blue)](https://github.com/AbdullahTabassam/Feature_Mapping_ORB_Computer_Vision)

---

For any inquiries or collaboration opportunities, please feel free to contact me.

**Author**: Abdullah Ikram Ullah Tabassam

<a href="https://www.linkedin.com/in/abdullah-ikram-ullah-tabassam-1103b021b/" target="_blank" >Linkedin</a>

Email: <a href="mailto:abdullahdar2017@gmail.com" >abdullahdar2017@gmail.com</a>
