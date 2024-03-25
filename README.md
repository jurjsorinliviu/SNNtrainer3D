# SNNtrainer3D
Training Spiking Neural Networks using a User-friendly Application with 3D Architecture Visualization Capabilities

This document outlines the features and functionality of the **SNNtrainer3D: Training Spiking Neural Networks using a User-friendly Application with 3D Architecture Visualization Capabilities** software application.
## Overview
The Spiking Neural Network Trainer with Network Visualization is a software application designed to train Spiking Neural Networks (SNNs). It provides a user-friendly interface for designing and training models, with the ability to visualize the model architecture using Three.js. Users can add, remove, and edit hidden layers (currently, only fully connected layers are supported).

The SNNtrainer3D offers several novel features that make it a valuable tool for researchers and practitioners in designing neural networks. Its unique capabilities include:

1\. **Dynamic Architecture Editing**: The ability to add, remove, and edit hidden layers in real-time provides model experimentation and optimization flexibility. This feature allows researchers to quickly iterate on different network architectures, leading to more efficient and effective models.

2\. **Visualization with Three.js**: The [Three.js](https://threejs.org/) visualization of the model architecture is a novel feature that enhances model understanding. Users can inspect the connections and structure with a 3D network representation, facilitating deeper insights into the model's behavior. We decided to use the three.js cross-browser 3D JavaScript library and API because it enabled us to create and display 3D graphics directly in the web browser using WebGL (a low-level API for rendering 3D graphics on the web).

3\. **Integration with Memristor Technology**: The SNNtrainer3D lays the groundwork for future integration with physical memristors. Software tools like this will be crucial for designing and optimizing physical neural networks built using memristors (e.g., neuromorphic circuits). By training and visualizing SNNs in software, researchers can better understand how to implement these networks using physical memristors. The saved weights can be used to initialize memristor-based networks physically or in simulation.
## Features
\- Design and visualize SNN architectures.

\- Add, remove, and edit hidden layers.

\- 4 different types of neurons (LIF, Lapique, Synaptic, and Alfa)

\- Fully customizable network architecture

\- Integrated with the MNIST dataset

\- Train the model on the MNIST dataset.

\- Progress bar for training.

\- Download trained weights.

\- Plot the loss and accuracy of the model during training.
## Visualization
The software uses [Three.js](https://threejs.org/) to create a 3D visualization of the designed model architecture. This visualization enhances understanding of the model's structure and connections between layers (the user can zoom in, rotate, move, etc.). 

Here, the connections between layers are presented in two colors: green, the positive weights, and red, the negative weights. The intensity of the color is the absolute value of the weight. Before training, the color of the weights serves only for a better view of the weights' complexity in the SNN architecture; however, once training is done, the color of the weights will be updated, and the user can visualize the final weight colors that contributed to the resulted accuracy on the specific dataset it was trained on (e.g., MNIST).

**Note**: the weights could be updated automatically in real-time during training as well; however, due to the huge number of weights, performance will be impacted considerably, and for this reason, we decided to update them only once training is done (once the weights are downloaded, the user could use them directly in another script for inference or continue further training). However, in a future LTSpice implementation of an SNN, we should choose the real-time visualization of the weight changes.
## Setup
To set up and get started with the SNNtrainer3D, follow these steps:

1\. **Install Dependencies**: Ensure you install Python 3 (e.g., version 3.12) on your system. Use pip to install the required packages listed in the “requirements.txt” document as follows:

*pip install -r requirements.txt*

2\. **Run the Application**: Execute the “app.py” file using the Windows Terminal or your favorite Python IDE (e.g., Visual Studio Code) to run the Flask application, as follows:

*python app.py*

3\. **Access the Application GUI**: Once the Flask application runs, open your web browser and go to the provided link (typically `http://localhost:5000`) to access the SNNtrainer3D software. We implemented our software GUI using the Flask Python backend framework because it is lightweight and simple, allowing us to prototype research ideas faster. It is also a popular framework among researchers for building web applications.
## Usage
1\. **Design Model**: Customize the neural network architecture by adding, removing, or editing hidden layers.

2\. **Download Dataset**: Click the "Download MNIST Dataset" button to download and prepare the MNIST dataset for training.

3\. **Train Model**: Once the dataset is downloaded, click the "Train Model" button to begin training the model on the dataset.

4\. **Download Weights**: After the training process, click the "Download Weights" button to download the trained weights.
# Development Process
## 1\. Understanding the Requirements
The development process began with:

\- Designing SNN architectures.

\- Adding, removing, and editing hidden layers.

\- Downloading the MNIST dataset.

\- Training the model.

\- Downloading trained weights.

\- Visualizing the model architecture throughout the process.
## 2\. Choosing the Technology Stack
Given the interactivity of the application, we chose the following technology stack:

\- **Python**: Backend development using Flask for server-side logic and neural network training.

\- **HTML/CSS/JavaScript**: Frontend development for the user interface.

\- **Three.js**: Utilized for 3D visualization of the model architecture.
## 3\. Implementation
**Backend Development (Flask)**

\- Set up Flask application structure.

\- Created routes for handling dataset download, model training, and weight download.

**Deep Learning Module**

\- Implemented logic for dataset management, model training, and weight saving using PyTorch and snnTorch

**Frontend Development**

\- Designed user interface mockups to visualize the SNN architecture.

\- Developed HTML/CSS for the user interface components.

\- Implemented JavaScript functionality for dynamic interaction and visualization.

**Three.js Integration**

\- Integrated Three.js library to render 3D visualizations of the SNN architecture.

\- Translated SNN architecture data into Three.js-compatible format.

\- Utilized Three.js capabilities to create an intuitive and interactive visualization.
## 4\. Testing and Debugging
Extensive testing was conducted throughout the development process to ensure functionality and usability.
# Visualization Advantages
**Importance of Visualization**

\- SNNs can be complex and difficult to understand without visualization.

\- Visualizing the architecture aids in comprehending the structure and connections between neurons and layers.

\- Enhances user experience and facilitates model design and debugging.

\- SNNs mostly lack visualization tools, so incorporating one adds significant value.

\- Originality stems from the need to simplify understanding SNNs and provide a user-friendly interface for designing and training models.
## Future Features
\- **Support for Different Layer Types**: Enhance the flexibility of model design by adding support for various types of hidden layers, such as convolutional, recurrent, and pooling layers.

\- **Real-time Training Visualization**: Provide a real-time visualization of the training process, including metrics like loss and accuracy.

\- **Integration with Additional Datasets**: Incorporate functionality to download and use additional datasets beyond MNIST for training and evaluation.

\- **Bulk addition/editing of layers**: Users can add multiple layers simultaneously, with the option to specify the number of layers and their properties.
## Conclusions
The SNNtrainer3D provides a comprehensive solution for designing, training, and visualizing SNN models. Its intuitive interface and visualization capabilities make it a valuable tool for researchers and practitioners in artificial intelligence, specifically neuromorphic computing.



