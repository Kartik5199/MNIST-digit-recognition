


# Handwritten Digit Recognition (MNIST)
```markdown
A deep learning project to recognize handwritten digits (0–9) using a **Convolutional Neural Network (CNN)** trained on the **MNIST dataset**.  
The model achieves **99.24% validation accuracy** and uses **Image Data Augmentation** for improved generalization.
```
## Problem Statement
```
Handwritten digit recognition is a fundamental computer vision problem that involves correctly identifying digits from images of handwritten numbers.  
It has practical applications in postal mail sorting, bank check processing, digitizing forms, and automated grading systems.
```

## Features
```
- Recognizes digits (0–9) from images.
- Achieves **~99% accuracy** on validation data.
- **Image augmentation** to improve performance on varied handwriting styles.
- Web interface for uploading and predicting digits.
- Built with **TensorFlow / Keras** for model training.
- Supports **real-time prediction** through OpenCV preprocessing.
```
## Project Structure

```

├── app.py                # Flask/Streamlit app for prediction
├── model.py              # CNN model architecture and training
├── static/               # Static assets (backgrounds, CSS, images)
├── templates/            # HTML templates for frontend
├── model.h5              # Saved trained model
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```
## Technologies Used
```
- **Python**
- **TensorFlow / Keras**
- **OpenCV**
- **NumPy / Pandas**
- **Flask / Streamlit** (for web interface)
- **HTML / CSS / JavaScript**
```
## Results
```
| Metric | Value |
|--------|-------|
| **Training Accuracy** | 98.32% |
| **Validation Accuracy** | 99.24% |
| **Validation Loss** | ~0.02 |
```
### Observations:
```
- Performs exceptionally well on clean MNIST images.
- Struggles slightly with blurred or low-quality images.
- Data augmentation improved robustness against handwriting variations.
```
## How It Works
```
1. **Data Loading**: MNIST dataset from `keras.datasets`.
2. **Data Augmentation**: Rotation, zoom, and shift transformations using `ImageDataGenerator`.
3. **Model Architecture**:
   - 3 Convolutional Layers
   - MaxPooling & Dropout
   - Dense Layers with ReLU and Softmax
4. **Training**: Optimized using Adam optimizer and categorical cross-entropy loss.
5. **Prediction**: User uploads an image → Preprocessed → Model predicts digit.
```

## Installation & Usage
```
1. Clone the repository:
git clone https://github.com/yourusername/handwritten-digit-recognition.git
cd handwritten-digit-recognition
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python app.py
```

4. Open browser and go to:

```
http://127.0.0.1:5000
```

---

## Future Improvements

* Improve predictions for **blurred/low-quality images** using better preprocessing.
* Implement **GAN-based augmentation** for more diverse handwriting styles.
* Deploy model using **Docker** or **Heroku** for online access.
* Integrate with **mobile apps** for real-time digit recognition.

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

* **MNIST dataset** by Yann LeCun
* TensorFlow/Keras for deep learning framework

