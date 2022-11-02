FROM python:3.10

#Installing python requirements
COPY requirements.pip ./
pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.pip
RUN rm -f requirements.pip

RUN mkdir /bucket

WORKDIR /work
