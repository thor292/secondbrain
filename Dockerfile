# SecondBrain API (v2) for Cloud Run or any container host.
#
# One image, several commands:
#   default                                        -> the API (uvicorn on $PORT, 8080 on Cloud Run)
#   python -m secondbrain.schema                   -> create / update the database schema (run once per release)
#   python -m secondbrain.forecasting --under root -> nightly forecasts
#   python -m secondbrain.signals                  -> recalculate decision signals
#   python -m secondbrain.relationships            -> rebuild the relationship graphs (needs FalkorDB)
#
# All settings come from SB_* environment variables (see .env.example); no .env file is baked in.
# The Live Commerce simulator is not started here: in production, merchants send real data to /ingest.

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.18 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first, so code changes don't reinstall them.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY examples ./examples
RUN uv sync --frozen --no-dev

RUN useradd --system --uid 10001 --home /app app && chown -R app /app
USER app

ENV PATH=/app/.venv/bin:$PATH \
    PORT=8080 \
    SB_DEMO_MODE=false

EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn secondbrain.api:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
