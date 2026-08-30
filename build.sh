#!/usr/bin/env bash
# Exit on error
set -o errexit

# 1. Install python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 2. Pre-download and cache InsightFace buffalo_s models at build time so runtime is instant
python -c "
import os
os.environ['INSIGHTFACE_MODEL'] = 'buffalo_s'
from insightface.app import FaceAnalysis
print('[BUILD] Pre-downloading and preparing InsightFace buffalo_s model...')
app = FaceAnalysis(name='buffalo_s', allowed_modules=['detection', 'recognition'], providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))
print('[BUILD] InsightFace buffalo_s model pre-cached successfully.')
"
