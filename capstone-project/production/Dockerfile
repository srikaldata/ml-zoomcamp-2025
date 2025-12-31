# base image
FROM python:3.13-slim

# for uv virtual environment container
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# working directory
WORKDIR /code

ENV PATH="/code/.venv/bin:$PATH"

COPY "pyproject.toml" "uv.lock" ".python-version" ./

# sync the virtual env and install necessary dependencies
RUN uv sync --locked

# copy the raw data to train 
COPY data/ ./

# copy the pickled best model (OPTIONAL - it is not must, if you run the train.py in the container before serve.py)
COPY "pickled_model.bin" ./

# copy the dict vectorizer fitted on full training data (OPTIONAL - it is not must, if you run the train.py in the container before serve.py)
COPY "dictvec_fulltrain.bin" ./

# copy the training script
COPY "train.py" ./

# copy the serving script
COPY "serve.py" ./

# exposing the port of the container to serve the app
EXPOSE #

# using the serve script as the entrypoint and the exposed port
ENTRYPOINT ["uvicorn", "serve:app", "--host", "#.#.#.#", "--port", "#"]
