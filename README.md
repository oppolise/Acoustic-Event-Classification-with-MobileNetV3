# 🎵 Acoustic Event Classification with MobileNetV3

This project implements a real-time acoustic event classifier using MobileNetV3 and Streamlit. The model is trained to classify 10 different urban sound events using spectrograms from the UrbanSound8K dataset.

## ✨ Features

- Support for common image formats (jpg, jpeg, png)
- Interactive web interface built with Streamlit
- Classification probabilities for 10 urban sound categories

## 🎧 Supported Sound Categories

1. Air conditioner
2. Car horn
3. Children playing
4. Dog bark
5. Drilling
6. Engine idling
7. Gun shot
8. Jackhammer
9. Siren
10. Street music

## 📋 Requirements

- Python 3.8+
- PyTorch
- Streamlit

## 🚀 Installation

```bash
pip install -r requirements.txt
```

## 📱 Usage

1. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

2. Upload a spectrogram image of an audio signal (must be in jpg, jpeg, or png format)

3. Click the "Predict" button to see classification results

## 📄 Input Requirements

- Input must be a spectrogram image of an audio signal
- Supported image formats: JPG, JPEG, PNG
- Images will be automatically resized to 224x224 pixels

## 📊 Dataset

This model was trained on the UrbanSound8K dataset, which contains 8732 labeled sound excerpts of urban sounds from 10 classes. The spectrograms were generated from these audio files for training the visual classification model.

## 🤖 Model

The classifier uses MobileNetV3-Large architecture, fine-tuned for acoustic event classification through spectrogram analysis. The model was trained using PyTorch and achieves competitive accuracy on urban sound classification tasks.