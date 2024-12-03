# Use an official Python image as the base
FROM python:3.8-slim

# Set environment variables for Spark
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin

# Install system dependencies and the latest Java version
RUN apt-get update && apt-get install -y \
    default-jdk wget \
    && apt-get clean

# Dynamically set JAVA_HOME
RUN export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java)))) && \
    echo "JAVA_HOME=$JAVA_HOME" >> /etc/environment
ENV JAVA_HOME=/usr/lib/jvm/java-1.11.0-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin

# Install Spark and Hadoop
RUN wget https://archive.apache.org/dist/spark/spark-3.3.0/spark-3.3.0-bin-hadoop3.tgz \
    && tar -xvzf spark-3.3.0-bin-hadoop3.tgz -C /opt/ \
    && mv /opt/spark-3.3.0-bin-hadoop3 /opt/spark \
    && rm spark-3.3.0-bin-hadoop3.tgz

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy project files
WORKDIR /app
COPY . /app

# Set entrypoint for running the prediction script via spark-submit
ENTRYPOINT ["spark-submit", "--master", "local[*]", "/app/Wine_Prediction_Distributed_System_Apache_Spark/notebook/Wine_Prediction_standalone.py"]
