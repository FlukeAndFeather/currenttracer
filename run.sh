#!/usr/bin/env bash
uvicorn currenttracer.server:app --reload --port 8000 --env-file .env
