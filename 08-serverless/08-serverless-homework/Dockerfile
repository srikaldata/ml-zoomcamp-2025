FROM public.ecr.aws/lambda/python:3.13

RUN pip install onnxruntime pillow torchvision

COPY clothing-model-new.onnx .
COPY lambda_function.py

CMD ["lambda_function.lambda_handler"]


# in a new cli to build the docker image:
# docker build -t <docker image new name> .

# in a new cli to run the docker container:
# docker run -it --rm -p ####:#### <built docker image name>