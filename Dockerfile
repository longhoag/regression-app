# base python image
FROM python:3.10-slim

# set the working directory
WORKDIR /app

# install dependencies
RUN apt-get update && apt-get install -y gcc curl && rm -rf /var/lib/apt/lists/*

# copy and install requirements
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# copy application files from host
COPY . .

# expose fastapi port
EXPOSE 8000

# run api
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
