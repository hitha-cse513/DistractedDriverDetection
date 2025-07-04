# Distracted Driver Detection: AI-Powered Driver Behavior Classifier

## Overview

The **Distracted Driver Detection** project is a Flask-based web application that uses deep learning to identify and classify distracted driving behaviors. It leverages the **VGG16 Convolutional Neural Network (CNN)** architecture through **Keras and TensorFlow** to deliver accurate real-time predictions from uploaded driver images.

This tool is aimed at enhancing road safety by detecting actions such as texting, phone usage, or attentive driving, providing immediate feedback via an intuitive interface.

---

## Source

* The model is built on **VGG16 CNN architecture**, pre-trained on ImageNet and fine-tuned for driver distraction classification.
* The app uses **Flask** for the web interface, allowing users to upload images and receive real-time predictions.
* The training dataset is the **State Farm Distracted Driver Detection dataset**, which includes categorized images of drivers engaged in various activities.

---

## Features

* 📸 **Upload Driver Images**: Supports uploading driver images in JPG, JPEG, PNG formats.
* 🤖 **Deep Learning-Based Classification**: Uses a fine-tuned VGG16 CNN model to identify distractions.
* ⚡ **Real-Time Prediction**: Instant feedback on uploaded driver images.
* 🖥️ **Flask Web App**: Easy-to-use and lightweight web interface for image uploads and classification.
* 📊 **Awareness & Safety Insights**: Educates users on risky driving behaviors and promotes safer habits.
* ♿ **Accessibility Features**: Interface designed for ease of use by all users, including those with disabilities.

---

## Key Modules

### 1. Driver Distraction Classification

* Detects behaviors like:

  * 📱 Texting  
  * ☎️ Talking on the phone (left/right hand)  
  * 🙆 Adjusting hair or makeup  
  * 🍔 Eating or drinking  
  * 🛑 Safe driving (no distraction)

### 2. User-Friendly Interface

* Clean and intuitive design.
* Upload and classify images in seconds.

### 3. Real-Time Feedback

* Instant classification results.
* Helps users understand and correct unsafe behaviors.

---

## Installation & Usage

### 🔧 Clone the Repository

```bash
git clone https://github.com/your-username/distracted-driver-detection.git
cd distracted-driver-detection
```

## Installation & Usage

**### 📦 Install Dependencies  **
Ensure Python 3.7+ is installed, then run:

```bash
pip install -r requirements.txt
```
### 🧠 Model Weights  
Make sure the pre-trained model file `model_vgg16.h5` is available in the project directory.

If not, you can:

- Train the model using the provided Jupyter notebook.  
- Or download pre-trained weights trained on the [State Farm Driver Dataset](https://www.kaggle.com/c/state-farm-distracted-driver-detection).

Place the `model_vgg16.h5` file in the root of the project directory.

---

### ▶️ Run the App

```bash
python app.py
```
Once the server starts, open your browser and go to:

```bash
http://127.0.0.1:5000
```

Use the web interface to upload images and view predictions in real time.

---

## Screenshots

<img src="https://github.com/your-username/distracted-driver-detection/blob/main/screenshots/interface.png" alt="Web Interface Screenshot">  
<br>  
<img src="https://github.com/your-username/distracted-driver-detection/blob/main/screenshots/prediction.png" alt="Prediction Example">

---

## Dataset

Download the dataset from Kaggle:  
[State Farm Distracted Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection)

If you wish to retrain the model:

- Extract the dataset into a folder named `dataset/` at the project root.

---

## Acknowledgments

- Pretrained VGG16 Model from [Keras Applications](https://keras.io/api/applications/vgg/)
- Dataset: [State Farm Distracted Driver Detection - Kaggle](https://www.kaggle.com/c/state-farm-distracted-driver-detection)
- Flask for backend web interface

---

## License

This project is open-source and free to use under the [MIT License](LICENSE).


