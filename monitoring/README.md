# Monitoring Stack

Prometheus + Grafana for benchmarking metrics visualization.

## Quick Start

```bash
# Start monitoring services
./scripts/start_monitoring.sh

# Or manually:
cd monitoring
docker-compose up -d

# View:
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin/admin)
```

## Configuration

- `prometheus.yml`: Prometheus scrape config (targets localhost:8000)
- `datasources/prometheus.yml`: Grafana Prometheus datasource
- `dashboards/smoke-dashboard.json`: Smoke test dashboard
- `dashboards/dashboards.yml`: Grafana provisioning config

## Cleanup

```bash
docker-compose down -v
```

## Notes

The benchmark application (running on the host) exports metrics to `http://localhost:8000/metrics` (Prometheus format).
Prometheus scrapes this endpoint every 5 seconds.
