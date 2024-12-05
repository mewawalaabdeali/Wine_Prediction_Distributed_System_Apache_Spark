# Use an official Python image as the base
FROM python:3.8-slim

# Set environment variables for Spark
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin

# Set JAVA_HOME dynamically and install system dependencies including Java
RUN apt-get update && apt-get install -y \
    default-jdk wget curl \
    && export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java)))) && \
    echo "JAVA_HOME=$JAVA_HOME" >> /etc/environment && \
    echo "PATH=$JAVA_HOME/bin:$PATH" >> /etc/environment && \
    apt-get clean

ENV JAVA_HOME /usr/lib/jvm/java-17-openjdk-amd64
ENV PATH $JAVA_HOME/bin:$PATH

# Install Spark
RUN wget https://archive.apache.org/dist/spark/spark-3.3.0/spark-3.3.0-bin-hadoop3.tgz \
    && tar -xvzf spark-3.3.0-bin-hadoop3.tgz -C /opt/ \
    && mv /opt/spark-3.3.0-bin-hadoop3 /opt/spark \
    && rm spark-3.3.0-bin-hadoop3.tgz

# Create working directory
WORKDIR /Wine_Prediction_Distributed_System_Apache_Spark

# Copy only requirements.txt first to leverage Docker's layer caching for dependencies
COPY requirements.txt /Wine_Prediction_Distributed_System_Apache_Spark/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy Python scripts and notebooks
COPY notebook/Wine_Quality_docker.py /Wine_Prediction_Distributed_System_Apache_Spark/notebook/

# Copy the models directory
COPY models /Wine_Prediction_Distributed_System_Apache_Spark/models/

# Set entrypoint for running the prediction script via spark-submit
ENTRYPOINT ["spark-submit", "--master", "local[*]", "/Wine_Prediction_Distributed_System_Apache_Spark/notebook/Wine_Quality_docker.py"]
