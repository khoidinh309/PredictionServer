# Use the official Python base image with version 3.10.14
FROM python:3.10.14-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container at /app
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . .

# Set the environment variable to ensure the output of the print statements is displayed in the Docker logs
ENV PYTHONUNBUFFERED=1

# Expose the port that the Flask app will run on
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]