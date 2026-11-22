# Fully reproducible Linux environment (x86_64; mediapipe has no Linux ARM wheel).
#   docker build --platform linux/amd64 -t echogesture .
#   docker run --rm --platform linux/amd64 -v "$PWD/clips:/clips" echogesture python predict.py /clips/x.mp4
FROM --platform=linux/amd64 python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt constraints.txt ./
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt
COPY . .
RUN python scripts/doctor.py
CMD ["python", "scripts/doctor.py"]
