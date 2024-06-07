import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (256, 256))
    image = image / 255.0
    image = cv2.GaussianBlur(image, (5, 5), 0)
    return image

def segment_lungs(image):
    _, image = cv2.threshold(image, 0.5, 1, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return image

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
                labels.append(1 if folder == 'cancer_images' else 0)
    return np.array(images), np.array(labels)

def segment_lungs_batch(images):
    segmented_images = []
    for image in images:
        segmented_image = segment_lungs(image)
        segmented_images.append(segmented_image)
    return np.array(segmented_images)

def create_cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def classify_images_cnn(images, labels):
    # Reshape the images to the correct input shape for the model
    images = images.reshape(-1, 256, 256, 1)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    # Create and compile the CNN model
    model = create_cnn_model()
    
    # Train the CNN model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Accuracy: {accuracy}")
    
    # Predict the probabilities of the test set
    probabilities = model.predict(X_test)
    
    # Convert probabilities to class labels
    y_pred = np.round(probabilities).flatten()
    display_classification_metrics(y_test, y_pred)
    # Visualize the confusion matrix
    visualize_confusion_matrix(y_test, y_pred)
    
    # Select a specific test image to visualize its saliency map
    # For demonstration, let's use the first image in the test set
    test_image = X_test[0]
    
    # Convert the test image to a TensorFlow tensor and ensure it has the correct shape
    test_image_tensor = tf.convert_to_tensor(test_image, dtype=tf.float32)
    test_image_tensor = tf.expand_dims(test_image_tensor, axis=0) # Add batch dimension
    
    # Visualize the saliency map for the selected test image
    visualize_saliency_map(model, test_image_tensor, 1) # For class 1
    
    return accuracy
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

def visualize_saliency_map(model, image, class_index):
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss = prediction
    
    gradients = tape.gradient(loss, image)
    pooled_gradients = tf.reduce_mean(gradients, axis=(1, 2))
    saliency_map = tf.reduce_mean(tf.multiply(pooled_gradients, image), axis=-1)
    saliency_map = tf.maximum(saliency_map, 0) / tf.math.reduce_max(saliency_map)
    
    # Remove the batch dimension to make the saliency map compatible with imshow
    saliency_map = np.squeeze(saliency_map.numpy())
    
    # Check if the saliency map has non-zero values
    print("Unique values in saliency map:", np.unique(saliency_map))
    
    plt.figure(figsize=(10, 10))
    plt.imshow(saliency_map, cmap='viridis')
    plt.title(f"Saliency Map for Class {class_index}")
    plt.show()

# Example usage
data_dir = 'Data/train/'
images, labels = load_and_preprocess_images(data_dir)
segmented_images = segment_lungs_batch(images)
accuracy = classify_images_cnn(segmented_images, labels)
print(f"Accuracy: {accuracy}")


