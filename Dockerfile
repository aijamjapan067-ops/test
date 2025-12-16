# 1. ベースイメージとして公式のPythonイメージを使用
FROM python:3.11-slim

# 2. コンテナ内の作業ディレクトリを設定
WORKDIR /app

# 3. 依存関係のファイルを先にコピーしてインストール
#    (ソースコードより先にインストールすることで、Dockerのビルドキャッシュを有効活用する)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. アプリケーションのソースコードを作業ディレクトリにコピー
COPY . .

# 5. アプリケーションがリッスンするポートをコンテナの外部に公開
EXPOSE 8000

# 6. アプリケーションを起動するコマンド
#    GunicornをUvicornワーカー(4つ)と共に使用し、本番環境でアプリを実行
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8000"]
