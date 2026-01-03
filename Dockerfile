# 1. Use a lightweight Python image
FROM python:3.9-slim

# 2. Install the C++ compilers needed for dlib and face_recognition
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# 3. Set the directory in the container
WORKDIR /app

# 4. Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your code into the container
COPY . .

# 6. Start the app
CMD ["python", "app.py"]
