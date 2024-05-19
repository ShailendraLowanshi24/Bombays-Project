import joblib
import numpy as np
import cv2
from skimage.feature import hog, local_binary_pattern
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the trained models and preprocessing components
scaler = joblib.load('saved_models/ens_scaler.pkl')
pca = joblib.load('saved_models/ens_pca.pkl')
ensemble_clf = joblib.load('saved_models/ens_classifier.pkl')

# Function to preprocess a single image
def preprocess_image(image):
    # Resize image
    image = cv2.resize(image, (150, 150))
    # Convert to gray-scale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply histogram equalization
    equalized_image = cv2.equalizeHist(gray_image)
    return equalized_image

# Function to extract combined features from a single image
def extract_features(image):
    # Histogram features
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()

    # Sobel edge detection
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5).flatten()
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5).flatten()

    # HOG features
    hog_feature, _ = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)

    # LBP features
    lbp = local_binary_pattern(image, P=8, R=1, method='uniform')
    lbp_hist = np.histogram(lbp, bins=np.arange(0, 27), range=(0, 26))[0]

    # Combine all features
    combined_features = np.hstack((hist, sobelx, sobely, hog_feature, lbp_hist))
    return combined_features

# Mapping numerical labels to class names
class_names = ['Building', 'Sea', 'Mountains', 'Streets', 'Glacier', 'Forest']

# Function for inference
def classify_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        return "Error: Could not read the image."

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Extract features
    features = extract_features(preprocessed_image)

    # Scale the features
    scaled_features = scaler.transform([features])

    # Apply PCA
    pca_features = pca.transform(scaled_features)

    # Perform classification
    prediction = ensemble_clf.predict(pca_features)

    # Map numerical label to class name
    predicted_class = class_names[prediction[0]]

    return predicted_class

# Flask route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Flask route for image classification
@app.route('/classify', methods=['POST'])
def classify_image_route():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file temporarily
    file_path = 'temp.jpg'
    file.save(file_path)

    # Perform classification
    prediction = classify_image(file_path)

    return jsonify({"class": prediction})

if __name__ == '__main__':
    app.run(debug=True)
