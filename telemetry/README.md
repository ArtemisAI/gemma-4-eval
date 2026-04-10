# Telemetry Setup

Monitoring stack for llama-server Gemma4 inference: **Prometheus** (metrics) + **Loki** (logs) + **Grafana** (dashboards).

## Architecture

```
zion-training (llama-server)          artemisai-local (monitoring)
┌──────────────────────┐              ┌─────────────────────────┐
│ llama-server :8080   │──metrics──>  │ Prometheus :9090        │
│   /metrics endpoint  │              │   scrapes every 15s     │
│                      │              │                         │
│ Promtail :9080       │──logs────>   │ Loki :3100              │
│   reads journald     │              │   stores & indexes logs │
└──────────────────────┘              │                         │
                                      │ Grafana :3000           │
                                      │   dashboards & alerts   │
                                      └─────────────────────────┘
```

## Setup Steps

### 1. Prometheus (already running on artemisai-local)

Add the llama-server scrape target:

```bash
# Append to /etc/prometheus/prometheus.yml under scrape_configs:
sudo cat telemetry/prometheus/llama-server-scrape.yml >> /etc/prometheus/prometheus.yml
sudo systemctl reload prometheus
```

Copy alert rules:
```bash
sudo cp telemetry/prometheus/alert_rules.yml /etc/prometheus/rules/llama-server.yml
sudo systemctl reload prometheus
```

### 2. Loki (install on artemisai-local)

```bash
# Install Loki
sudo dnf install -y loki  # Fedora
# OR download binary: https://github.com/grafana/loki/releases

# Deploy config
sudo mkdir -p /etc/loki /var/lib/loki
sudo cp telemetry/loki/loki-config.yaml /etc/loki/config.yaml

# Create systemd service
sudo tee /etc/systemd/system/loki.service << 'EOF'
[Unit]
Description=Grafana Loki
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/loki -config.file=/etc/loki/config.yaml
Restart=on-failure
User=loki

[Install]
WantedBy=multi-user.target
EOF

sudo useradd -r -s /sbin/nologin loki 2>/dev/null
sudo chown -R loki:loki /var/lib/loki
sudo systemctl daemon-reload
sudo systemctl enable --now loki
```

### 3. Promtail (install on zion-training)

```bash
# On zion-training:
sudo apt install -y promtail  # Ubuntu
# OR download binary

sudo mkdir -p /etc/promtail /var/lib/promtail
sudo cp telemetry/loki/promtail-config.yaml /etc/promtail/config.yaml
sudo systemctl enable --now promtail
```

### 4. Grafana Dashboard

Import the dashboard JSON:

```bash
# Via Grafana API
curl -X POST http://admin:admin@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @telemetry/grafana/dashboards/llama-server.json
```

Or import manually: Grafana UI > Dashboards > Import > Upload JSON.

## Available Metrics

| Metric | Type | Description |
|---|---|---|
| `llamacpp:prompt_tokens_total` | counter | Total prompt tokens processed |
| `llamacpp:tokens_predicted_total` | counter | Total generation tokens |
| `llamacpp:predicted_tokens_seconds` | gauge | Current generation speed (t/s) |
| `llamacpp:prompt_tokens_seconds` | gauge | Current prompt processing speed |
| `llamacpp:requests_processing` | gauge | Active requests |
| `llamacpp:requests_deferred` | gauge | Queued requests |
| `llamacpp:n_decode_total` | counter | Total decode calls |

## Useful Queries

```promql
# Generation speed over time
llamacpp:predicted_tokens_seconds{instance="zion-training"}

# Tokens generated per minute
rate(llamacpp:tokens_predicted_total[5m]) * 60

# Request throughput
rate(llamacpp:n_decode_total[5m]) * 60
```

### Loki Queries (LogQL)

```logql
# All llama-server logs
{job="llama-server"}

# Errors only
{job="llama-server"} |= "error" or "Error" or "ASSERT"

# Slot completions with speed
{job="llama-server"} |= "speed:"
```
