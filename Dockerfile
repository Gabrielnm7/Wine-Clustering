# I just testing the dockerization. It is not ready yet so 
# Dissmiss this

FROM python:3.9.5

WORKDIR /fastapi-test

COPY requirements.txt .
COPY pipeline.py .
RUN pip install -r requirements.txt

CMD ["python", "pipeline.py"]