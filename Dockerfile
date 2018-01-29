FROM python:3.5

MAINTAINER Dmitry Ustalov <dmitry.ustalov@gmail.com>

COPY . /usr/src/app

WORKDIR /usr/src/app

RUN pip --no-cache-dir install -r requirements.txt
