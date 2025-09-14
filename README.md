# Image Classifier Web App

A full-stack machine learning web application for real-time image classification, built with Flask, PyTorch, and scikit-learn.

## Features

- **Deep Learning Integration:** Uses a pretrained ResNet-18 model from PyTorch for general image classification on 1,000+ ImageNet classes.
- **Custom ML Pipeline:** Includes a Random Forest classifier trained on the scikit-learn Digits dataset for handwritten digit recognition.
- **Modern Web UI:** Responsive frontend with custom CSS and Jinja2 templates for seamless user experience.
- **File Uploads:** Secure image upload and preprocessing pipeline using Pillow and NumPy.
- **Production-Ready:** Modular codebase, easy to extend for new models or datasets.

## Tech Stack

- **Backend:** Flask, PyTorch, scikit-learn, joblib, Pillow, NumPy
- **Frontend:** HTML5, CSS3, Jinja2 templates
- **Model Training:** Random Forest (Digits), ResNet-18 (ImageNet)

## Usage

1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/image-classifier-webapp.git
    cd image-classifier-webapp
    ```

2. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Train the digit model (optional):**
    ```sh
    python train_model_test.py
    ```

4. **Run the app:**
    - For general image classification:
        ```sh
        python app.py
        ```
    - For handwritten digit classification:
        ```sh
        python app_digits.py
        ```

5. **Open your browser:**  
   Navigate to `http://localhost:5000` and upload an image to get instant predictions.

## Code Highlights

- [app.py](app.py): Flask app for ResNet-18 predictions ([`predict`](app.py))
- [app_digits.py](app_digits.py): Flask app for digit recognition ([`predict`](app_digits.py))
- [train_model_test.py](train_model_test.py): Model training pipeline ([`RandomForestClassifier`](train_model_test.py))
- [templates/index.html](templates/index.html): Upload form UI
- [templates/results.html](templates/results.html): Prediction results UI
