FROM agrigorev/zoomcamp-model:2025 as base

FROM agrigorev/zoomcamp-model:2025

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /code

ENV PATH="/code/.venv/bin:$PATH"

COPY "pyproject.toml" "uv.lock" ".python-version" ./

RUN uv sync --locked

COPY "question6_predict.py" ./

EXPOSE #

ENTRYPOINT ["uvicorn", "question6_predict:app", "--host", "#.#.#.#", "--port", "#"]
