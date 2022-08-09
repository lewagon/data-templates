FROM python:3.8.12-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# You can add --server.port $PORT if you need to set PORT as a specific env variable
CMD streamlit run app.py
