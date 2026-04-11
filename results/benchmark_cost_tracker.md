# Benchmark Cost Tracker
## Session: 2026-04-10

### Attempt 1 (destroyed — SSH perms broken on nvidia/cuda image)
| Instance | GPU | ID | $/hr | Status |
|----------|-----|----|------|--------|
| 4090-bench | RTX 4090 | 34554185 | $0.28 | DESTROYED — ~5 min, cost ~$0.02 |
| 5090-bench | RTX 5090 | 34554188 | $0.35 | DESTROYED — ~5 min, cost ~$0.03 |

### Attempt 2 (active — pytorch/pytorch image)
| Instance | GPU | ID | $/hr | Start Time | End Time | Duration | Cost |
|----------|-----|----|------|------------|----------|----------|------|
| 4090-bench | RTX 4090 | 34554586 | $0.38 | 2026-04-10 ~now | - | - | - |
| 5090-bench | RTX 5090 | 34554588 | $0.38 | 2026-04-10 ~now | - | - | - |
| **3090 (local)** | RTX 3090 | zion-training | $0.00 | local | local | - | $0.00 |

**Sunk cost from attempt 1:** ~$0.05
**Combined rate:** $0.76/hr while both instances running
**Budget target:** minimize — provision, benchmark, destroy
**Estimated benchmark time:** ~30-45 min → **est. total cost: $0.43-$0.62**
