# ใช้ Python 3.11 official slim image เป็นฐาน
FROM python:3.11-slim

# ตั้งค่าระบบการทำงานของ Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# กำหนดโฟลเดอร์เริ่มต้นการทำงานในคอนเทนเนอร์
WORKDIR /app

# ติดตั้งไลบรารีระบบและสร้างไดเรกทอรีสำหรับจัดเก็บไฟล์แคชของโมเดล
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /app/.cache

# กำหนดให้ระบบจัดเก็บโมเดลภาษาไทยไว้ในโฟลเดอร์แคชเพื่อทำ Volume Mount
ENV HF_HOME=/app/.cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache

# คัดลอกและติดตั้งข้อกำหนดของระบบไลบรารี
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกโค้ดและทรัพยากรทั้งหมดเข้าคอนเทนเนอร์
COPY . .

# เปิดพอร์ตการเชื่อมต่อสำหรับ Streamlit
EXPOSE 8501

# กำหนดคำสั่งเริ่มต้นสำหรับเปิดใช้งานระบบเว็บแอปพลิเคชัน
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
