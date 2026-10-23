FROM python:3.11.9-slim

WORKDIR /app

# 3. Tránh Python ghi file .pyc và bật log xuất trực tiếp ra terminal
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 4. Copy file khai báo thư viện vào trước để tận dụng Docker Cache
COPY requirement.txt .

# 5. Nâng cấp pip và cài đặt các thư viện cần thiết
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirement.txt

COPY . .