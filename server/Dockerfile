FROM python:3.9.10

RUN apt update

RUN apt install -y python3-opencv libavdevice-dev libavfilter-dev libopus-dev libvpx-dev pkg-config libsrtp2-dev

RUN mkdir /app

WORKDIR /app

ADD requirements.txt /app

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

ADD . /app

ENTRYPOINT ["gunicorn", "main:app", "--bind", ":8080", "--worker-class", "aiohttp.GunicornWebWorker"]
