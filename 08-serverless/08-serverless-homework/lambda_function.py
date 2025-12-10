# importing and viewing the model
import onnxruntime as ort

onnx_model_path = 'hair_classifier_v1.onnx'

session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])


# # QUESTION 1 - Name of the output
# get the inputs and outputs
inputs = session.get_inputs()
outputs = session.get_outputs()


# the name of the first input and output
input_name = inputs[0].name
output_name = outputs[0].name


# The name of the output of the given model is 'output'

# # QUESTION 2 - Downloading image and preparing it for output
from io import BytesIO
from urllib import request

from PIL import Image

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

# importing numpy
import numpy as np

# importing preprocessing modules
from torchvision import transforms


# image preprocessor
def img_preprocessor(temp_url):
    # FETCHING THE IMAGE:
    # downloading the image
    temp_img_url = temp_url
    temp_img = download_image(temp_img_url)

    # the target input image size needs to be
    target_size = (200, 200)
    
    # Based on the previous homework, the target size for the image needs to be image size 200 x 200
    # image prepped
    temp_img_resized = prepare_image(temp_img, target_size=target_size)

    # PREPROCESSING
    # normalization values for rgb values after totensor transformation
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # image transformations for training and test set
    img_transform = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # loading the images
    temp_ppimg = img_transform(temp_img_resized)
    
    return temp_ppimg.unsqueeze(0).numpy()


# # # QUESTION 4 - Applying the model to the image and finding the output
# result = session.run([output_name], {input_name: temp_ppimg.unsqueeze(0).numpy()})

# predictions = result[0][0].tolist()

# print('Image prediction probability:', predictions)


# lambda handler function that runs for each request
def lambda_handler(event, context):
    url = event['url']
    X = img_preprocessor(url)
    prediction = session.run([output_name], {input_name: X})
    return prediction[0]
