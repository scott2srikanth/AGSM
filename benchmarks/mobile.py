"""Mobile measurements require an on-device runner and identified hardware."""

MOBILE_MEASUREMENT_FIELDS = (
    "device_model",
    "os_version",
    "runtime_version",
    "precision",
    "threads",
    "streaming_chunk_size",
    "median_ms",
    "p95_ms",
    "peak_rss_bytes",
    "energy_per_1000_tokens",
)
