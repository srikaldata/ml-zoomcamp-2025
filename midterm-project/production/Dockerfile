FROM python:3.13-slim

# necessary files needed for uv virtual environment container
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# setting the working directory
WORKDIR /code

ENV PATH="/code/.venv/bin:$PATH"

COPY "pyproject.toml" "uv.lock" ".python-version" ./

RUN uv sync --locked

# copy the raw data to train 
COPY data/ ./

# copy the pickled best model (optional - if you run the train.py in the container, it is not needed)
COPY "pickled_model.bin" ./

# copy the dict vectorizer fitted on full training data (optional - if you run the train.py in the container, it is not needed)
COPY "dictvec_fulltrain.bin" ./

# copy the train script
COPY "train.py" ./

# copy the predict script
COPY "predict.py" ./

# exposing the port of the container to serve the app
EXPOSE #

ENTRYPOINT ["uvicorn", "predict:app", "--host", "#.#.#.#", "--port", "#"]
