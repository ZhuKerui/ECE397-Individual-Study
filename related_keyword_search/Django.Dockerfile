FROM python:3.7
ENV PYTHONUNBUFFERED 1
WORKDIR /code
ADD requirements.txt requirements_cp.txt
RUN pip install -r requirements_cp.txt