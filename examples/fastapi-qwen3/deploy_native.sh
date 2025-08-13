#!/bin/bash
uvicorn multirequest_model_server:app --host 0.0.0.0 --port 8000 --workers 1
