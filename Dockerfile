FROM python:3.7-stretch

WORKDIR /usr/src/PAI

COPY . .

RUN pip install -r requirements.txt

ENV PYTHONPATH=$PYTHONPATH:/usr/src/PAI
