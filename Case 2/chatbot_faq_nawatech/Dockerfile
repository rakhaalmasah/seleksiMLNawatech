# Gunakan image Python slim
FROM python:3.10-slim

# Set direktori kerja
WORKDIR /app

# Salin dan install dependensi
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh file proyek ke image
COPY . .

# Jalankan aplikasi Streamlit
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
