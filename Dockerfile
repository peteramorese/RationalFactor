FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

WORKDIR /root/RationalFactor

RUN apt-get -y update && apt -y install git

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install --no-cache-dir -e .
