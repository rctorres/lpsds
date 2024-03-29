FROM python:3.10

#Installing vim (git)
RUN apt update
RUN apt install --yes vim

#Installing python requirements
COPY requirements.pip ./
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.pip
RUN rm -f requirements.pip

RUN mkdir /bucket

WORKDIR /work
