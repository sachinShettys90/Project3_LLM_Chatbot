FROM python:3.12-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "Chatbot_Langgraph_Frontend_Langsmith.py", "--server.address=0.0.0.0", "--server.port=8501"]
