FROM python:3.9

WORKDIR /app
COPY . .

RUN pip install numpy==1.23.5
RUN pip install -r requirements.txt

ENV PORT=8080

CMD gunicorn --bind 0.0.0.0:$PORT robo_dashboard:server