FROM python:3.12-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8090

CMD ["uvicorn","mcwRAGAPIBedrock:app","--host","0.0.0.0","--port","8090"]