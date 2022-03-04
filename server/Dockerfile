FROM michaelfong2017/tensorrt22.02-py3-opencv4.5.3:latest

RUN apt update

RUN apt install -y libavdevice-dev libavfilter-dev libopus-dev libvpx-dev pkg-config libsrtp2-dev

RUN mkdir /app

WORKDIR /app

ADD requirements.txt /app

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

ADD . /app

CMD ["python", "main.py"]