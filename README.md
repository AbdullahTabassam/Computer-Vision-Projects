# Computer Vision Projects Repository

Welcome to my Computer Vision Projects Repository. This repository contains a collection of projects that I have worked on in the field of computer vision. Each project demonstrates my skills and experience in applying computer vision techniques to solve real-world problems. 

## Contents

1. [In-flight Bird Species Detection and Counting](https://github.com/AbdullahTabassam/In-flight_Bird_Species_Detection_Counting_)
2. [3D Reconstruction Project](https://github.com/AbdullahTabassam/3D-Recontruction)
3. [Lung Tumor Detection using YOLOv8 Segmentation Models and Flask Web Application](https://github.com/AbdullahTabassam/Lungs_Tumor_Segmentation_Computer_Vision)
4. [Object Detection Model for Bird Species Identification](https://github.com/AbdullahTabassam/Custom-Object-Detection)
5. [UK Garden Birds Detection using YOLO v8 model](https://github.com/AbdullahTabassam/Yolo_Bird_detection/)
6. [Project 6](#project-6)
7. [Project 7](#project-7)
8. [Project 8](#project-8)
9. [Project 9](#project-9)

---

### Project 1: In-flight Bird Species Detection and Counting

Welcome to the repository for the In-flight Bird Species Detection and Counting project! This project, developed as part of an MSc thesis, focuses on utilizing object detection models for the categorization, species detection, and counting of birds. By leveraging computer vision techniques, this project aims to contribute to wildlife conservation and research efforts.

#### Demo
##### Video 
Check out our project in action in the video demo: [Video Demo](https://www.linkedin.com/posts/abdullah-ikram-ullah-tabassam-1103b021b_computervision-birdconservation-innovationinscience-activity-7102575697251987456-6Z-r?utm_source=share&utm_medium=member_desktop)

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

**Author**: Abdullah Ikram Ullah Tabassam

<a href="https://www.linkedin.com/in/abdullah-ikram-ullah-tabassam-1103b021b/" target="_blank" >Linkedin</a>

Email: <a href="mailto:abdullahdar2017@gmail.com" >abdullahdar2017@gmail.com</a>

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

**Author**: Abdullah Ikram Ullah Tabassam

<a href="https://www.linkedin.com/in/abdullah-ikram-ullah-tabassam-1103b021b/" target="_blank" >Linkedin</a>

Email: <a href="mailto:abdullahdar2017@gmail.com" >abdullahdar2017@gmail.com</a>

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

### Video

Check out our project in action in the video demo: [Video Demo](https://media.licdn.com/dms/image/D4D05AQHNh6ho3QT69Q/videocover-high/0/1692633680506?e=1719612000&v=beta&t=nmIfXSKzF8bGCjYnZcL-lNNjEF896-ObEhicp-VlsW4)


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
---

For any inquiries or collaboration opportunities, please feel free to contact me.

**Author**: Abdullah Ikram Ullah Tabassam

<a href="https://www.linkedin.com/in/abdullah-ikram-ullah-tabassam-1103b021b/" target="_blank" >Linkedin</a>

Email: <a href="mailto:abdullahdar2017@gmail.com" >abdullahdar2017@gmail.com</a>
