# 🎯**Reuseable Gaze Representation Learning from Videos**


**Authors:** Datao Liang & Rohan Bhattaram
**Promoter:** Bace Mihai  
**Supervisor:** Alexandre Personnic  

## 📌 **Overview**  
This project focuses on learning **reusable gaze representations** from video data using **deep learning models**. Our goal is to develop a **generalizable gaze estimation model** that works across different tasks, such as **biometrics, human-computer interaction, and medical diagnostics**.

Our model integrates:  
  - **ResNet as the backbone** for feature extraction  
  - **Adaptive Group Normalization (AdaGN)** for multi-modal balance  
  - **Attention Mechanisms** (Channel & Self-Attention) for feature refinement  
  -  **Global Average Pooling (GAP)** to prevent overfitting  
  - **GRU (Gated Recurrent Units)** for capturing **temporal gaze dependencies**  

## 📂 Dataset  
We use the **EVE Dataset** for training, which includes:  
  - **Multi-view gaze recordings** from different screen setups  
  - **Eye, face, and screen data** as separate inputs  
  - **Head position (x, y, z), head orientation (yaw, pitch, roll), and gaze direction vectors**  

## 🏗️ Model Architecture  
  - **Feature Extraction (ResNet)**: Extracts spatial features from **eye, face, and screen data**.  
  - **Feature Fusion & Normalization (AdaGN)**: Ensures balanced feature contributions.  
  - **Attention Mechanisms (Channel & Self-Attention)**: Learns feature importance and spatial dependencies.  
  - **Global Average Pooling (GAP)**: Reduces dimensions to prevent overfitting.  
  - **GRU for Temporal Processing**: Captures gaze movement over time.  
  - **Fully Connected (FC) Layer**: Maps features to **(x, y) gaze coordinates**.  

