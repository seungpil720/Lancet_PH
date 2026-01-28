# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.9-slim

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install production dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the local code to the container image.
COPY . .

# Cloud Run expects the container to listen on port 8080.
EXPOSE 8080

# Run the Streamlit app.
# --server.port=8080 tells Streamlit to listen on the port Cloud Run expects.
# --server.address=0.0.0.0 tells it to listen on all interfaces.
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
