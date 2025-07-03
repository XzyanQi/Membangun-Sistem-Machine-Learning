from prometheus_client import start_http_server, Counter, Gauge
import time

metric = Counter("metric_name", "desc")

start_http_server(8001)

while True:
    metric.inc()
    time.sleep(1)
