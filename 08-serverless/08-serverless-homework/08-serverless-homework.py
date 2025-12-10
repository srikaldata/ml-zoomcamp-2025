#!/usr/bin/env python


# install onnx runtime locally 
# %pip install onnxruntime



# to download images using PIL package
# %pip install pillow



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

print('Input name:', input_name)

print('Output name:', output_name)


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



# downloading the image
temp_img_url = 'https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg'

temp_img = download_image(temp_img_url)

print(temp_img)



# the target input image size needs to be
target_size = (200, 200)


# importing numpy
import numpy as np


temp_img_resized = prepare_image(temp_img, target_size=target_size)

print(temp_img_resized)


# Based on the previous homework, the target size for the image needs to be image size 200 x 200

# # QUESTION 3 - Value of Red in first pixel after pre-processing the image


# importing preprocessing modules
from torchvision import transforms



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
print(temp_ppimg)

# the shape of the RGB channeled image sized 200x200 pixels
print(temp_ppimg.shape)

# finding the first pixel in the R channel
# R channel [index 0] --> first row [index 0] --> first column [index 0]
print(temp_ppimg[0][0][0])


# The first pixel's red channel has a value of **-1.0733** after usual image preprocessing

# # QUESTION 4 - Applying the model to the image and finding the output


print(temp_ppimg.unsqueeze(0).numpy().shape)

result = session.run([output_name], {input_name: temp_ppimg.unsqueeze(0).numpy()})

predictions = result[0][0].tolist()

print('Image prediction probability:', predictions)


# The output of the model: 0.09156

# # QUESTION 5 - Size fo the base image

# Converting the notebook to a script by running
# 
# jupyter nbconvert --to script 08-serverless-homework.ipynb --output-dir="08-serverless-homework"

