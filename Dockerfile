# 1. Use an official lightweight Python base image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy requirements file first (for efficient Docker caching)
COPY requirements.txt .

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy application code and trained model into container
COPY main.py .
COPY churn_model.pkl .

# 6. Expose port 8000 for network access
EXPOSE 8000

# 7. Command to launch Uvicorn web server when container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]