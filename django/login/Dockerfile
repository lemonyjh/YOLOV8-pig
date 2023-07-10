FROM python:3.6

RUN cd /opt/ &&\
        pip install django &&\
        django-admin startproject docker &&\
    cd docker/ &&\
    django-admin startapp aplication

WORKDIR /opt/docker
EXPOSE 8080
CMD python manage.py runserver 0.0.0.0:8080
