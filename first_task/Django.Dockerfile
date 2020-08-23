FROM python:3.7
ENV PYTHONUNBUFFERED 1
RUN mkdir /code
RUN mkdir /root/nltk_data
WORKDIR /code
ADD requirements.txt requirements_cp.txt
RUN pip install -r requirements_cp.txt
COPY . /code/
COPY nltk_data /root/nltk_data