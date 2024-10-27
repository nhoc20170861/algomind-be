#!/bin/bash

# Mitigate database
alembic upgrade head

# Chạy project FastAPI
uvicorn main:app --host 0.0.0.0 --port 8000 --reload