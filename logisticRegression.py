
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import numpy as np
import os
import cv2
from skimage.feature import greycomatrix, greycoprops
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize the image
    image = cv2.resize(image, (256, 256))
    
    # Normalize the image
    image = image / 255.0
    
    # Apply Gaussian blur to reduce noise
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    return image


def extract_features(image):
    # Convert the image to an unsigned 8-bit integer type
    image = np.uint8(image * 255)
    
    # Calculate texture features using GLCM (Gray Level Co-occurrence Matrix)
    glcm = greycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = greycoprops(glcm, 'contrast')
    dissimilarity = greycoprops(glcm, 'dissimilarity')
    homogeneity = greycoprops(glcm, 'homogeneity')
    energy = greycoprops(glcm, 'energy')
    correlation = greycoprops(glcm, 'correlation')
    
    # Calculate statistical features
    mean = np.mean(image)
    std_dev = np.std(image)
    
    # Combine all features
    features = np.hstack([contrast.ravel(), dissimilarity.ravel(), homogeneity.ravel(), energy.ravel(), correlation.ravel(), mean, std_dev])
    
    return features


def segment_lungs(image):
    # Thresholding to segment the lungs
    _, image = cv2.threshold(image, 0.5, 1, cv2.THRESH_BINARY)
    
    # Morphological operations to remove small noise
    kernel = np.ones((5, 5), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    
    return image
############################################################################################################################################

def load_and_preprocess_images(data_dir):
    images = []
    labels = []
    
    for folder in ['normalTest', 'cancer_images']:
        folder_path = os.path.join(data_dir, folder)
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(folder_path, filename)
                image = preprocess_image(image_path)
                images.append(image)
                labels.append(1 if folder == 'cancer_images' else 0) # 1 for cancer, 0 for normal
    
    return np.array(images), np.array(labels)

data_dir = 'Data/train/'
images, labels = load_and_preprocess_images(data_dir)
def segment_lungs_batch(images):
    segmented_images = []
    for image in images:
        segmented_image = segment_lungs(image)
        segmented_images.append(segmented_image)
    return np.array(segmented_images)

segmented_images = segment_lungs_batch(images)

def extract_features_batch(segmented_images):
    features = []
    for image in segmented_images:
        feature = extract_features(image)
        features.append(feature)
    return np.array(features)

features = extract_features_batch(segmented_images)

def visualize_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
def display_classification_metrics(y_test, y_pred):
    # Calculate precision, recall, F1-score, and support for each class
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average=None)
    
    # Print the metrics for each class
    print("Classification Report:")
    for p, r, f, s in zip(precision, recall, fscore, support):
        print(f"Class {np.argmax(support):d}: Precision={p:.2f}, Recall={r:.2f}, F1-Score={f:.2f}, Support={s:d}")


def classify_images(features, labels):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Train a Random Forest Classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    
    # Predict the labels of the test set
    y_pred = clf.predict(X_test)

    importances = clf.coef_[0]
    print(importances)
    feature_names = ['contrast','dissimilarity','homogeneity','energy','correlation','mean','standard_deviation']
    plt.figure(figsize=(10, 5))
    plt.bar(feature_names, importances)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importances')
     # Rotate x-axis labels for better readability
    plt.show()
    
    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)
    visualize_confusion_matrix(y_test, y_pred)
    display_classification_metrics(y_test, y_pred)
    return accuracy

accuracy = classify_images(features, labels)
print(f"Accuracy: {accuracy}")