FROM python:3.11-slim-buster

RUN apt update && apt install ffmpeg libsm6 libxext6  -y

ARG USER_ID
ARG GROUP_ID

RUN groupadd -g ${GROUP_ID} cont_user && \
    useradd -l -u ${USER_ID} -g cont_user cont_user && \
    install -d -m 0755 -o cont_user -g cont_user /home/cont_user

USER cont_user

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

WORKDIR /app/
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt