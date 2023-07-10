FROM python:3

ENV PYTHONUNBUFFERED 1

RUN mkdir /code

COPY requirements.txt /code/

RUn pip install -i https:pypi.doubanio.com/simple/ -r requirements.txt

COPY ./code/

COPY start.sh /code/start.sh

RUN chmod +x /code/start.sh

ENTPYPOINT cd /code; ./start.sh

EXPOSE 8080