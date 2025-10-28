FROM agrigorev/zoomcamp-model:2025

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /code

ENV PATH="/code/.venv/bin:$PATH"

COPY "pyproject.toml" "uv.lock" ".python-version" ./

RUN uv sync --locked

COPY "question4_predict.py" "pipeline_v1.bin" ./

EXPOSE #

ENTRYPOINT ["uvicorn", "question4_predict:app", "--host", "#.#.#.#", "--port", "#"]
