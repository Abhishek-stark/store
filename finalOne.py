from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Button

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()


class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Classifier")

        # Set the size of the window
        self.root.geometry("800x600")

        self.label = Label(root, text="Select an image to classify")
        self.label.pack(pady=20)

        self.upload_button = Button(root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10)

        self.image_label = Label(root)
        self.image_label.pack(pady=10)

        self.result_label = Label(root, text="", font=('Helvetica', 16))
        self.result_label.pack(pady=20)

    def upload_image(self):
        # Open a file dialog to select an image file
        file_path = filedialog.askopenfilename()

        if file_path:
            # Read and display the image using OpenCV
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

            # Convert the image to Tkinter format
            image = ImageClassifierApp.convert_to_tkinter_format(image)

            self.image_label.config(image=image)
            self.image_label.image = image

            # Predict the class and confidence score
            self.predict_image(file_path)

    def predict_image(self, file_path):
        # Read the image file
        image = cv2.imread(file_path)

        # Resize the image to the model's input size
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

        # Make the image a numpy array and reshape it to the model's input shape
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

        # Normalize the image array
        image = (image / 127.5) - 1

        # Predict the model
        prediction = model.predict(image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Display the prediction result
        result_text = f"Class: {class_name[2:].strip()} | Confidence Score: {str(np.round(confidence_score * 100))[:-2]}%"
        self.result_label.config(text=result_text)

    @staticmethod
    def convert_to_tkinter_format(image):
        # Convert the image from OpenCV format to Tkinter format
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.imencode('.png', image)[1].tostring()
        return tk.PhotoImage(data=image)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassifierApp(root)
    root.mainloop()
