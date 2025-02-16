# Face-Detector

![output](https://github.com/user-attachments/assets/2dae3f03-bd0a-43fe-8f52-08706b9f6b70)

This project presents a **Face Detector** designed for **detecting faces in images and associating them with previously seen identities**. The system is built upon a **VGGFace2 model**, fine-tuned using **ArcFace** and **Triplet** Losses to enhance recognition performance.  

The implementation provides:  
- A **face bank** for storing and managing known identities.  
- An **application** capable of detecting faces from images or a live webcam feed.  

The project leverages a **deep learning-based algorithm** implemented with the **PyTorch** framework, achieving **97% classification accuracy** on a subset of the CelebA dataset.

---

## Table of Contents  
- [Overview](#overview)  
- [Datasets](#datasets)
- [System Workflow](#system-workflow)  
- [Technologies Used](#technologies-used)  
- [Results](#results)  
- [Contributors](#contributors)
  
---

## Overview  
The goal of this project was to create a system capable of accurately detecting and recognizing faces, The main functionalities include:  
- Identifing and locating faces in images or video streams.
- Aligning faces for the neural network.
- Match detected faces against a database of known identities.
- Managing known faces for recognition.
- Performing detection and recognition on live webcam input. 
  
---

## Datasets  

- **Training:** The model was trained using the **CelebA** dataset.  
- **Testing & Evaluation:** A subset of CelebA was used for testing and performance metrics.  
- **Presentation:** A custom **Players Dataset** was created for demonstration purposes, containing a few images of **Kylian Mbappé** and **Cristiano Ronaldo**.  

⚠ **Disclaimer:** I do not claim ownership of the images used in the *Players Dataset*. They are included solely for demonstration purposes.  

---

## System Workflow  
![DLF drawio](https://github.com/user-attachments/assets/7b474345-b120-4ddd-b65f-081ecf90aa1f)

---

## Technologies Used  
- **Python 3**
- **PyTorch** - for building and training deep learning models
- **OpenCV** – for image processing and feature detection  
- **NumPy** – for efficient numerical operations  
- **Matplotlib** – for visualization  
- **einops** – for tensor manipulation  
- **Git/GitHub** – for version control and collaboration

---

## Results  

The face recognition system achieved **97% classification accuracy** on a subset of the **CelebA dataset**, demonstrating high reliability in identifying known faces.  

To further evaluate performance, we analyzed **False Rejection Rate (FRR)** and **False Acceptance Rate (FAR)** across different decision thresholds:  

- **FRR (False Rejection Rate)** – Measures the percentage of genuine faces incorrectly rejected by the system.  
- **FAR (False Acceptance Rate)** – Measures the percentage of impostor faces incorrectly accepted as known identities.  

The following figure presents the **FRR and FAR curves**, showing the trade-off between security and recognition accuracy:  

![output](https://github.com/user-attachments/assets/1e307a47-2218-41ae-ac92-967ccf52a77d)

A well-balanced threshold ensures both **low rejection of genuine users** and **high resistance to false acceptances**.  

---

## Contributors  
- **Jakub Bednarski** – Conceptualization, Methodology, Software Development, Project Administration  
- **Julia Komorowska** – Software Development, Investigation
- **Adam Wasiela** – Software Development, Investigation  
- **Hubert Woziński** – Software Development, Investigation  
