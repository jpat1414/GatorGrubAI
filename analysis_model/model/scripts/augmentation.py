# This script is used to create a transformation pipeline for image augmentation.

import torchvision.transforms.v2 as transforms
from PIL import Image

# loading ImageNet base 
from torchvision.models import vgg16
from torchvision.models import VGG16_Weights

from langchain.llms import HuggingFacePipeline
import easyocr
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms.v2 as transforms
from PIL import Image
import easyocr
import cv2
from huggingface_hub import InferenceClient

# Client Initialization
client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.3")

keywords = ['free', 'food', 'pizza', 'burger', 'sandwich']

def apply_augmentation(image_path):
    """
    Applies augmentation and OCR to image, uses mistral model for classification.
    """
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 4)

    # Resize the image
    scale_percent = 200  # Scale image 2x
    width = int(thresh.shape[1] * scale_percent / 100)
    height = int(thresh.shape[0] * scale_percent / 100)
    resized = cv2.resize(thresh, (width, height), interpolation=cv2.INTER_LINEAR)

    # Convert to PIL Image for further transformations
    pil_image = Image.fromarray(resized)

    # Apply torchvision transformations
    IMG_width, IMG_height = 224, 224
    transformation = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.Grayscale(num_output_channels=3),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.RandomResizedCrop(size=(IMG_width, IMG_height), scale=(0.9, 1.0), ratio=(1, 1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard ImageNet normalization
    ])
    augmented_image = transformation(pil_image)

    # Perform OCR
    reader = easyocr.Reader(['en'])
    results = reader.readtext(resized)
    detected_text = [result[1].lower() for result in results]

    # Use Hugging Face Client for classification
    response = client.text_generation(f'''The text is: {detected_text}.
ANSWER ONLY 'Free food detected' or 'No free food detected'
if any of the words in the text are in the keywords list: {keywords}.
Do not return anything else.
Pizza can look like pizal, but it is still pizza.''')

    return augmented_image, response


'''
def process_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshed = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return threshed

processed_image = process_image("images.png")
cv2.imwrite("processed_image.jpg", processed_image)

results = reader.readtext(processed_image)
detected_text_post_processing = [result[1].lower() for result in results]'



post_process = [kw for kw in keywords if any(kw in text for text in detected_text_post_processing)]

if post_process:
    print("Free Food Detected After Processing:" + post_process)
else:
    print("No free food detected even after processing.")'
'''