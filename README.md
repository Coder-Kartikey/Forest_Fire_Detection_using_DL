# Forest Fire Detection using Deep Learning

This repository contains a deep learning project for detecting forest fires from images using Convolutional Neural Networks (CNNs) built with TensorFlow and Keras. The objective is to classify images as either "fire" or "nofire" to help in early and automated detection of wildfires.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Prediction on New Images](#prediction-on-new-images)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Project Overview

Forest fires are a major environmental hazard, often resulting in severe ecological, economic, and human losses. Early detection is crucial to minimize their impact. This project implements a CNN-based classifier that can distinguish between images of forests with and without fires.

The project covers:
- Data loading and preprocessing
- Data augmentation and visualization
- Model building, training, and evaluation
- Model saving and making predictions on new images

## Dataset

The project uses the [The Wildfire Dataset](https://www.kaggle.com/datasets/elmadafri/the-wildfire-dataset) available on Kaggle. It contains images categorized into two classes: `fire` and `nofire`, split into training, validation, and test sets.

**Folder Structure:**
```
the_wildfire_dataset_2n_version/
├── train/
│   ├── fire/
│   └── nofire/
├── val/
│   ├── fire/
│   └── nofire/
└── test/
    ├── fire/
    └── nofire/
```

## Model Architecture

The model is a simple CNN built using TensorFlow Keras:

- **Input Layer:** 150x150x3 RGB images
- **Convolutional Layers:** 3 Conv2D layers with increasing filters (32, 64, 128), each followed by MaxPooling
- **Flatten Layer**
- **Dense Layer:** 512 units + Dropout
- **Output Layer:** 1 unit, sigmoid activation (binary classification)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Coder-Kartikey/Forest_Fire_Detection_using_DL.git
   cd Forest_Fire_Detection_using_DL
   ```

2. **Install required Python packages:**
   ```bash
   pip install tensorflow numpy matplotlib kagglehub
   ```

3. **Download the dataset:**
   The code uses `kagglehub` to fetch the dataset. Make sure you have Kaggle API credentials set up.

## Usage

All the code is in the Jupyter notebook: `Forest_Fire_Detection_using_DL.ipynb`.

### Training

- Run the notebook from start to finish to:
  - Download and explore the dataset
  - Visualize sample images
  - Build and train the CNN model
  - Evaluate performance on validation and test sets

### Predicting on New Images

After training, the model is saved as `FFD.keras`. You can use the provided `predict_fire(img_path)` function to predict whether a new image contains a forest fire.

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

model = load_model('FFD.keras')

def predict_fire(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    predicted_class = 'nofire' if prediction[0] <= 0.5 else 'fire'
    plt.imshow(img)
    plt.title(f'Predicted: {predicted_class}')
    plt.axis('off')
    plt.show()
```

## Results

- **Validation Accuracy:** ~78% (after 12 epochs)
- **Test Accuracy:** ~80%
- Training and validation accuracy/loss curves are plotted in the notebook.

## Acknowledgements

- [Kaggle: The Wildfire Dataset](https://www.kaggle.com/datasets/elmadafri/the-wildfire-dataset)
- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)


---

**Contributions and feedback are welcome!**

# Owner
CoderKP
