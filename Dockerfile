# Step 1: Use a Python base image
FROM python:3.9-slim

# Step 2: Install system dependencies required for dlib and OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Step 3: Set the working directory
WORKDIR /app

# Step 4: Copy requirements first (to leverage Docker caching)
COPY requirements.txt .

# Step 5: Install Python dependencies
# This is where pymongo and dlib get installed
RUN pip install --no-cache-dir -r requirements.txt

# Step 6: Copy the rest of your application code
COPY . .

# Step 7: Expose the port your app runs on (usually 8080 for IBM)
EXPOSE 8080

# Step 8: Start the application using Gunicorn for production stability
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]