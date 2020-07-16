FROM python:3.7-stretch

WORKDIR /usr/src/PAI

COPY . .

RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple  -r requirements.txt

ENV PYTHONPATH=$PYTHONPATH:/usr/src/PAI:$PATH
