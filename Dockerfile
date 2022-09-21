FROM python:3.10

WORKDIR /work

#Installing python requirements
COPY requirements.pip ./
RUN pip install --no-cache-dir -r requirements.pip
RUN rm -f requirements.pip

RUN mkdir /bucket
