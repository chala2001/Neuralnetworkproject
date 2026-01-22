import tensorflow as tf
import os
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('skin_type_model.h5')

# Define the class labels (the order should match the folder structure in your training data)
class_labels = ['dry', 'normal', 'oily']

# Function to preprocess and predict on a new image
def predict_skin_type(img_path):
    try:
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(128, 128))  # Match the size used during training
        img_array = image.img_to_array(img)  # Convert the image to a numpy array
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Rescale the pixel values to [0, 1]

        # Make the prediction
        predictions = model.predict(img_array)

        # Get the predicted class (index with highest probability)
        predicted_class_index = np.argmax(predictions)

        # Map the predicted class index to the corresponding label
        predicted_class = class_labels[predicted_class_index]

        return predicted_class

    except Exception as e:
        print(f"Error processing the image {img_path}: {e}")
        return None

# Function to iterate over all images in a directory and predict the skin type for each
def predict_on_all_images_in_folder(train_dir):
    correct_predictions = 0
    total_predictions = 0

    # Iterate over each folder (dry, oily, normal) inside the train directory
    for class_name in class_labels:
        class_folder = os.path.join(train_dir, class_name)
        
        if os.path.isdir(class_folder):
            # Iterate over each image in the class folder
            for img_name in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_name)
                
                if img_name.lower().endswith(('png', 'jpg', 'jpeg')):
                    # Get the predicted skin type for the image
                    predicted_class = predict_skin_type(img_path)

                    if predicted_class is None:
                        continue  # Skip if prediction failed

                    # Increment total predictions
                    total_predictions += 1

                    # Check if the predicted class matches the actual class
                    if predicted_class == class_name:
                        correct_predictions += 1
                    
                    # Print the image name and predicted class
                    print(f"Image: {img_name} | Predicted Label: {predicted_class}")

    # Calculate and print the accuracy
    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    print(f"\nAccuracy of the model: {accuracy:.2f}%")

# Set the path to your train directory
train_dir = r"C:\Users\ASUS\Desktop\Neuralnetworkproject\practical"

# Call the function to predict on all images in the train folder and print accuracy
predict_on_all_images_in_folder(train_dir)
