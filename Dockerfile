FROM python:3.10

ARG GITHUB_ACCESS_TOKEN

WORKDIR /work

#Installing docker (DIND)
RUN apt-get update
RUN apt-get --yes install ca-certificates curl gnupg lsb-release
RUN mkdir -p /etc/apt/keyrings
RUN curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
RUN apt-get update
RUN apt-get install --yes docker-ce docker-ce-cli containerd.io docker-compose-plugin

#Passing github authentication so we can install private github packages via pip.
RUN git config --global url."https://$GITHUB_ACCESS_TOKEN:@github.com/".insteadOf "https://github.com/"

#Installing python requirements
COPY requirements.pip ./
RUN pip install --no-cache-dir -r requirements.pip
RUN rm -f requirements.pip

RUN mkdir /bucket
