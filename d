version: '3.8'

services:
  db:
    image: postgres:13
    environment:
      POSTGRES_DB: your_database
      POSTGRES_USER: your_user
      POSTGRES_PASSWORD: your_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U your_database"]
      interval: 10s
      timeout: 5s
      retries: 5

  data_fetcher:
    build:
      context: .
      dockerfile: Dockerfile
    depends_on:
      db:
        condition: service_healthy
    environment:
      DB_NAME: your_database
      DB_USER: your_user
      DB_PASSWORD: your_password
      DB_HOST: localhost
      DB_PORT: 5432
    volumes:
      - .:/app
    command: ["python", "dynamic_intraday_data_fetch.py"]

  data_analyzer:
    build:
      context: .
      dockerfile: Dockerfile
    depends_on:
      db:
        condition: service_healthy
    environment:
      DB_NAME: your_database
      DB_USER: your_user
      DB_PASSWORD: your_password
      DB_HOST: localhost
      DB_PORT: 5432
    volumes:
      - .:/app
    command: ["sh", "-c", "wait-for-it localhost:5432 -- python Evidently-drift-detection.py"]

volumes:
  postgres_data: