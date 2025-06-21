# 🧠 Hand-Written-Digit-Recognition-using-MNIST-DATASET

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

📌 Overview

This project is focused on classifying handwritten digits using a dense neural network built with TensorFlow and Keras. The model is trained and evaluated on the MNIST dataset, which is 

a benchmark dataset for image classification in machine learning. The goal is to classify grayscale images of handwritten digits (0–9) into their 

respective categories.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

✨ Features

Simple and easy-to-understand neural network

Uses TensorFlow/Keras framework

Fully connected (dense) layers

Input normalization and reshaping

Visualization of sample images and confusion matrix

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🧱 Dense Layer Architecture

Input layer: Flattened 28x28 grayscale image to 784-dimensional vector

Hidden layer: Dense layer with 128 neurons and ReLU activation

Output layer: Dense layer with 10 neurons (one per digit) and Softmax activation

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

📊 Dataset: MNIST
Source: https://www.kaggle.com/datasets/hojjatk/mnist-dataset

Training Samples: 60,000

Testing Samples: 10,000

Image Size: 28x28 pixels

Channels: 1 (grayscale)

MNIST consists of 70,000 grayscale images of handwritten digits (0 to 9), each of size 28x28 pixels.

It is divided into 60,000 training images and 10,000 test images.

Each image is labeled with the correct digit it represents.

The pixel values range from 0 to 255 and are normalized to the range 0 to 1 before feeding into the model.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

📈 Results

Loss Function: Sparse Categorical Crossentropy

Optimizer: Adam

Epochs: 10

Batch Size: 32

Test Accuracy Achieved: 92%

Confusion Matrix:

![image](https://github.com/user-attachments/assets/588c9be4-4155-4eed-aed2-94273df49a2f)


-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

📚 References

https://www.kaggle.com/datasets/hojjatk/mnist-dataset

https://www.geeksforgeeks.org/machine-learning/handwritten-digit-recognition-using-neural-network/

https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

👨‍💻 Author

Muqadas Ejaz

BS Computer Science (AI Specialization)

Machine Learning & Computer Vision Enthusiast

📫 Connect with me on [LinkedIn](https://www.linkedin.com/in/muqadasejaz/)  
🌐 GitHub: [github.com/muqadasejaz]( https://github.com/muqadasejaz)

