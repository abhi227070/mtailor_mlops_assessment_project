# Use official Python image
FROM python:3.9-slim

# Set work directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

# Expose port (only for local testing if needed)
EXPOSE 8080

# Define entry point (must be `run` function in Cerebrium)
CMD ["python", "app.py"]
