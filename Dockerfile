# I just testing the dockerization. It is not ready yet so 
# Dissmiss this

FROM python:3.9

ADD pipeline.py .

RUN pip install requirement.txt

CMD ["python", "./pipeline.py"]