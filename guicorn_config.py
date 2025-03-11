import os

# Gunicorn configuration
workers = int(os.environ.get('GUNICORN_WORKERS', 4))
threads = int(os.environ.get('GUNICORN_THREADS', 2))
bind = "0.0.0.0:" + os.environ.get('PORT', '8000')
worker_class = 'gthread'
timeout = 120
keepalive = 5