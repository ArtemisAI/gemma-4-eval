"""System design eval tasks.

Architecture-level questions requiring capacity estimation, trade-off analysis,
and concrete technical decisions. Tests deep reasoning + structured output.
"""


def get_tasks() -> list[dict]:
    return [
        {
            "name": "distributed_cache_100m_dau",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Design a distributed caching layer for a social media platform with 100M DAU. "
                        "Requirements:\n"
                        "- 500M cache entries, average 2KB each\n"
                        "- 99.9th percentile read latency < 5ms\n"
                        "- Support cache-aside, write-through, and write-behind patterns\n"
                        "- Handle cache stampede / thundering herd on cold starts\n"
                        "- Cross-datacenter consistency (US-East, US-West, EU-West)\n\n"
                        "Provide: architecture diagram (text), capacity estimation with actual numbers, "
                        "data partitioning strategy, failure modes and mitigations, "
                        "and a concrete technology stack with justification for each choice."
                    ),
                }
            ],
        },
        {
            "name": "realtime_ml_feature_store",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Design a real-time ML feature store that serves features for a fraud detection "
                        "system processing 50,000 transactions per second. Requirements:\n"
                        "- Batch features (updated hourly): user spending patterns, merchant risk scores\n"
                        "- Streaming features (sub-second): transaction velocity, geo-anomaly detection\n"
                        "- Point-in-time correctness for training (no future data leakage)\n"
                        "- Feature versioning and lineage tracking\n"
                        "- P99 serving latency < 10ms\n\n"
                        "Provide concrete architecture, data flow, storage layer choices, "
                        "and explain how you handle the offline-online consistency problem."
                    ),
                }
            ],
        },
        {
            "name": "multi_region_event_sourcing",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Design an event-sourced system for a financial trading platform that operates "
                        "across 3 regions (NYC, London, Tokyo). Requirements:\n"
                        "- Exactly-once event processing guarantee\n"
                        "- Causal ordering of events within an account\n"
                        "- Eventual consistency across regions with conflict resolution\n"
                        "- Event replay capability for auditing and debugging\n"
                        "- Snapshot strategy for accounts with >100K events\n"
                        "- Regulatory compliance: immutable audit trail, GDPR right-to-erasure\n\n"
                        "Address the tension between immutable events and GDPR erasure. "
                        "Provide event schema examples, partition strategy, and failure scenarios."
                    ),
                }
            ],
        },
    ]
