FROM python:3.12-slim

# OpenCV runtime dependencies (libGL, libGLib)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy pipeline modules (imported by backend/tasks.py at runtime)
COPY calibration.py disparity.py pallet.py measurement.py \
     pipeline.py pointcloud.py config.py logging_setup.py ./

# Copy FastAPI backend package
COPY backend/ ./backend/

# Persistent data directory (sessions, calibrations, captures)
VOLUME /app/data

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
