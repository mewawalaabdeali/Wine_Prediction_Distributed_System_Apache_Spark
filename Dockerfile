# Use an official Python image as the base
FROM python:3.8-slim

# Set environment variables for Spark
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    && apt-get clean

# Install Spark and Hadoop
RUN wget https://archive.apache.org/dist/spark/spark-3.3.0/spark-3.3.0-bin-hadoop3.tgz \
    && tar -xvzf spark-3.3.0-bin-hadoop3.tgz -C /opt/ \
    && mv /opt/spark-3.3.0-bin-hadoop3 /opt/spark \
    && rm spark-3.3.0-bin-hadoop3.tgz

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the project files into the container
WORKDIR /app
COPY . /app

# Set entrypoint for running the prediction script via spark-submit
ENTRYPOINT ["spark-submit", "--master", "local[*]", "/app/Wine_Prediction_Distributed_System_Apache_Spark/notebook/Wine_Prediction_standalone.py"]
