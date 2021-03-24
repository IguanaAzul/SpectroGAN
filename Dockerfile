from python:3.7
RUN apt update -y && apt upgrade -y
RUN apt install libasound2-dev python-dev python-numpy python-setuptools libsndfile-dev ffmpeg libavcodec-dev -y
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
WORKDIR /code