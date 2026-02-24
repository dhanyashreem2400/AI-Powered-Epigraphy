# AI-Powered Epigraphy

This repository provides an AI-powered solution for epigraphy, focusing on detecting characters from inscriptions, analyzing character images, and predicting each charater in the output directory using supervised learning.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Step 1: Detect Characters](#step-1-detect-characters)
  - [Step 2: Perform Epigraphy Analysis](#step-2-perform-epigraphy-analysis)
- [Acknowledgments](#acknowledgments)

## Installation

To use this project, follow these steps:

1. **Clone the Repository**

    ```bash
    git clone https://github.com/dhanyashreem2400/ai-powered-epigraphy.git
    cd ai-powered-epigraphy
    ```

2. **Install Required Packages**

    Install the necessary Python packages i.e OpenCV, Numpy, Matplotlib, Tensorflow using pip:


3. **Install Tesseract**

   **Windows**: Download and install Tesseract from [this link](https://github.com/tesseract-ocr/tesseract). Add the installation path to your system’s environment variables.
    

## Usage

### Step 1: Detect Characters

1. **Select Your Image**

    Place your images containing inscriptions in the `static` directory.

2. **Run Detection Script**

    Execute the following command to detect characters and save them to the `output2_boxes` directory:

    ```bash
    python letter_detection.py 
    ```

    This script processes the image in the `static` directory, detects characters, and saves individual character images to the `output2_boxes` directory.

<p align = "center">
  <img src = "https://github.com/user-attachments/assets/bb8146b1-db9d-4f9c-943f-51f3fcbeda1e" width = "500">
</p>


### Step 2: Perform Epigraphy Analysis

1. **Train or Load Your Model**

    The model is trained using large datasets of [Devanagari Inscriptions](https://www.kaggle.com/datasets/ashokpant/devanagari-character-dataset) and it is named as [AI_Epigraphy.keras](AI_Epigraphy.keras)

2. **Run Analysis Script**

    Execute the following command to analyze and predict the class of each character image:

    ```bash
    python main.py 
    ```

    This script reads each image from the `output2_boxes` directory, uses the AI model to predict the class of each character, and outputs the predictions.

<p align = "center">
  <img src = "https://github.com/user-attachments/assets/92644fb3-639f-4259-b9c5-e1dc7cd60ae2" alt = "Predicted image" width = "400" height = "400">
</p>


## Acknowledgments

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for character recognition.
- [OpenCV](https://opencv.org/) for image processing.
- [Devanagari-Dataset](https://www.kaggle.com/datasets/ashokpant/devanagari-character-dataset) for providing excclusive dataset
