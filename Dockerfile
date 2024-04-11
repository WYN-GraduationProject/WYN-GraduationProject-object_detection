FROM python:3.11

LABEL authors="wangyaning"

ENV TZ=Asia/Shanghai

COPY requirements.txt /src/

WORKDIR /src
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8002

CMD ["python", "object_detection_server.py"]