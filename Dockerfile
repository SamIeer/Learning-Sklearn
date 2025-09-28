# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set a working directory in the container
WORKDIR /app

# Copy requirements if any (you may have to create this)
# If you have a requirements.txt in your repo, use that.
# Otherwise, you’ll install necessary packages manually.
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app

# Expose a port if needed (for example, for Dash, Notebook, etc.)
EXPOSE 8888

# Default command: you can adjust this to what you want to run
# For example, to start Jupyter Notebook:
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]
