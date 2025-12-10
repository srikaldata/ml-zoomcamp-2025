FROM agrigorev/model-2025-hairstyle:v1

# installing dependenices
RUN pip install onnxruntime pillow torchvision

# optional to copy the existing model
COPY hair_classifier_v1.onnx .

# copying the lambda function developed
COPY lambda_function.py

# calling the lambda_handler fn inside lambda_handler script
CMD ["lambda_function.lambda_handler"]


# in a new cli to build the docker image:
# docker build -t <docker image new name> .

# in a new cli to run the docker container:
# docker run -it --rm -p ####:#### <built docker image name>