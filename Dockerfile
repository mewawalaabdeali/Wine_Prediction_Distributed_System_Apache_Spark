# Use an official Python image as the base
FROM python:3.8-slim

# Set environment variables for Spark
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin

# Install system dependencies including the latest Java version
RUN apt-get update && apt-get install -y \
    default-jdk wget curl \
    && apt-get clean

# Set JAVA_HOME dynamically
RUN export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java)))) && \
    echo "JAVA_HOME=$JAVA_HOME" >> /etc/environment && \
    echo "PATH=$JAVA_HOME/bin:$PATH" >> /etc/environment
ENV JAVA_HOME /usr/lib/jvm/java-17-openjdk-amd64
ENV PATH $JAVA_HOME/bin:$PATH

# Install Spark
RUN wget https://archive.apache.org/dist/spark/spark-3.3.0/spark-3.3.0-bin-hadoop3.tgz \
    && tar -xvzf spark-3.3.0-bin-hadoop3.tgz -C /opt/ \
    && mv /opt/spark-3.3.0-bin-hadoop3 /opt/spark \
    && rm spark-3.3.0-bin-hadoop3.tgz

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy only the prediction file
WORKDIR /app
COPY notebook/Wine_Quality_docker.py /app/

# Set entrypoint for running the prediction script via spark-submit
ENTRYPOINT ["spark-submit", "--master", "local[*]", "/app/Wine_Quality_docker.py"]
