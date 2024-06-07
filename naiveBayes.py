import cv2
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skimage.feature import greycomatrix, greycoprops
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split


def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (256, 256))
    image = image / 255.0
    image = cv2.GaussianBlur(image, (5, 5), 0)
    return image

def extract_features(image):
    # Convert the image to an unsigned integer type
    image = np.uint8(image * 255)
    
    # Compute the GLCM
    glcm = greycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    
    # Extract texture features
    contrast = greycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = greycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
    energy = greycoprops(glcm, 'energy')[0, 0]
    correlation = greycoprops(glcm, 'correlation')[0, 0]
    mean = np.mean(image)
    std_dev = np.std(image)
    
    # Combine features into a single array
    features = np.array([contrast, dissimilarity, homogeneity, energy, correlation, mean, std_dev])
    return features

def load_and_preprocess_images(data_dir):
    images = []
    labels = []
    for folder in ['cancer_images', 'normal']:
        folder_path = os.path.join(data_dir, folder)
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(folder_path, filename)
                image = preprocess_image(image_path)
                features = extract_features(image)
                images.append(features)
                labels.append(1 if folder == 'cancer_images' else 0)
    return np.array(images), np.array(labels)

data_dir = 'Data/train/'
images, labels = load_and_preprocess_images(data_dir)


# Splitting the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

def visualize_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
# Implementing Naive Bayes Classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Predicting on the test set
y_pred = gnb.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)

confusion_matrix_result = visualize_confusion_matrix(y_test, y_pred)
feature_names = ['contrast','dissimilarity','homogeneity','energy','correlation','mean','std',]
# Assuming 'imp' is the result of permutation_importance
imp = permutation_importance(gnb, X_test, y_test, n_repeats=10, random_state=42)

# Sort the features by their importance
sorted_idx = imp.importances_mean.argsort()[::-1]

# Ensure feature_names matches the number of features
if len(feature_names) <= len(sorted_idx):
    # Use the sorted_idx to reorder feature_names if necessary
    feature_names = np.array(feature_names)[sorted_idx]
else:
    # If feature_names has more entries than sorted_idx, truncate it
    feature_names = feature_names[:len(sorted_idx)]

# Create a horizontal bar plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.barh(feature_names, imp.importances_mean[sorted_idx], align='center')
ax.set_xlabel('Permutation Importance')
ax.set_title('Permutation Importance of Features')

plt.show()

print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(confusion_matrix_result)



