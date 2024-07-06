# Feature Extraction in Construction Surface Crack Detection Systems

GitHub repository for my master's thesis at the University of Belgrade, School of Electrical Engineering, Department of Signals and Systems. 

This repository contains MATLAB codes for detecting surface cracks on images of concrete construction surfaces obtained from https://www.kaggle.com/datasets/arunrk7/surface-crack-detection.

Code structure:

**1. Feature Extraction and Analysis - uporedna_analiza_obelezja.m**

Tool for applying various feature extraction methods: statistical measures, morphological features, frequency domain features, and Hough transform-based features.
Feature analysis can be performed with different methods, such as box-plot analysis, cross-correlation, and histogram analysis.
Additionally, LDA and PCA can be performed.

**2. Support Vector Machine - SVM.m**

Script for applying support vector machines for classifying images based on selected features in three dimensions.

**3. Genetic Algorithm - GA.m**

Implementation of the genetic algorithm for a three-dimensional classification task. Genetic algorithm classification is achieved by defining the classification plane polynomial coefficients as the genetic algorithm variables.

**4. Kolmogorov-Arnold Networks - KAN.py**

Implementation of Kolmogorov-Arnold Networks, as proposed in https://github.com/KindXiaoming/pykan, for classifying three-dimensional features. Installation requirements listed in requirements.txt. This script contains the full training and inference pipeline.

**5. Convolutional Kolmogorov-Arnold Networks - KKAN.py & KKAN_classification.py**

Implementation of Convolutional Kolmogorov-Arnold Networks, as proposed in https://github.com/AntonioTepsich/Convolutional-KANs, for image classification. Installation requirements listed in KKAN_requirements.txt. KKAN.py contains the Convolutional KAN model, while KKAN_classification.py contains the full training and inference pipeline for KKAN and a traditional Convolutional Neural Network model proposed in ConvNet.py.
