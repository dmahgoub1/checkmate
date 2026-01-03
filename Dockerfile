# Use a stable Python base
FROM python:3.9-slim

# Prevent interactive prompts during the build process
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Step 1: Install Build Essentials
# Splitting these helps IBM cache the heavy system downloads
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Step 2: Install Math & X11 Libraries
# Required for dlib's linear algebra and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-dev \
    liblapack-dev \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# Step 3: Pre-install dlib
# We do this BEFORE copying your code so that changes to index.html 
# do not trigger a 20-minute dlib rebuild.
RUN pip install --no-cache-dir dlib face-recognition

# Step 4: Set up the Application Directory
WORKDIR /app

# Step 5: Copy requirements and install
# This handles any other libraries in your requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 6: Copy the rest of your project (including index.html)
COPY . .

# Step 7: Final Command
CMD ["python", "app.py"]