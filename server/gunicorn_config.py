import multiprocessing

# Number of workers = (2 x number of CPU cores) + 1
workers = (2 * multiprocessing.cpu_count()) + 1

# Bind to all interfaces
bind = "0.0.0.0:5000"

# Worker timeout
timeout = 120

# Keep-alive connections
keepalive = 5

# Logging
accesslog = "-"  # stdout
errorlog = "-"   # stdout
loglevel = "info"

# Worker class
worker_class = "sync"

# Maximum number of requests a worker will process before restarting
max_requests = 1000
max_requests_jitter = 50 