-- ============================================================
--  Predictive Maintenance — TimescaleDB Schema
--  Phase G2: Database Migration
--  Run: docker exec -i pm-timescaledb psql -U pm_user -d predictive_maintenance < server/deep_analysis/schema.sql
-- ============================================================

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ─── 1. Server Registry (Relational) ────────────────────────
-- Stores metadata about each patient server that registers via MQTT.
CREATE TABLE IF NOT EXISTS servers (
    server_id       TEXT PRIMARY KEY,
    server_name     TEXT,
    hostname        TEXT,
    os              TEXT,
    arch            TEXT,
    topology        JSONB,              -- full hardware_topology.json from auto_discovery
    registered_at   TIMESTAMPTZ DEFAULT NOW(),
    last_seen_at    TIMESTAMPTZ DEFAULT NOW()
);

-- ─── 2. Raw Telemetry (Hypertable) ──────────────────────────
-- Every telemetry tick from every patient server.
-- Auto-partitioned by time for fast range queries.
CREATE TABLE IF NOT EXISTS telemetry_raw (
    time            TIMESTAMPTZ NOT NULL,
    server_id       TEXT NOT NULL,
    schema_version  TEXT DEFAULT '1.0',
    payload         JSONB NOT NULL,     -- full hierarchical v2 telemetry JSON
    -- Extracted columns for fast indexed queries:
    cpu_temp        DOUBLE PRECISION,
    cpu_load        DOUBLE PRECISION,
    gpu_temp        DOUBLE PRECISION,
    gpu_power       DOUBLE PRECISION,
    ram_used_gb     DOUBLE PRECISION,
    ram_total_gb    DOUBLE PRECISION,
    disk_temp       DOUBLE PRECISION,
    health_score    DOUBLE PRECISION,
    fault_class     INTEGER,
    confidence      DOUBLE PRECISION
);

-- Convert to hypertable (auto-partitioned by time)
SELECT create_hypertable('telemetry_raw', 'time', if_not_exists => TRUE);

-- Index for fast server-specific queries
CREATE INDEX IF NOT EXISTS idx_telemetry_server_time
    ON telemetry_raw (server_id, time DESC);

-- ─── 3. AI Predictions Log (Hypertable) ─────────────────────
-- Every inference result from Edge and Cloud models.
CREATE TABLE IF NOT EXISTS predictions (
    time            TIMESTAMPTZ NOT NULL,
    server_id       TEXT NOT NULL,
    source          TEXT DEFAULT 'edge',   -- 'edge' or 'cloud'
    model_name      TEXT,                  -- 'classifier_int8', 'iso_forest', 'mad_gan', etc.
    fault_class     INTEGER,
    fault_label     TEXT,
    confidence      DOUBLE PRECISION,
    ttf_hours       DOUBLE PRECISION,
    anomaly_score   DOUBLE PRECISION,
    inference_ms    DOUBLE PRECISION,      -- latency in milliseconds
    shap_data       JSONB                  -- SHAP feature attributions
);

SELECT create_hypertable('predictions', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_predictions_server_time
    ON predictions (server_id, time DESC);

-- ─── 4. Adaptive Thresholds (per server, per metric) ────────
-- Computed by the threshold engine, pushed to Edge via MQTT.
CREATE TABLE IF NOT EXISTS thresholds (
    server_id       TEXT NOT NULL,
    metric_name     TEXT NOT NULL,
    warning_value   DOUBLE PRECISION,
    critical_value  DOUBLE PRECISION,
    p50_value       DOUBLE PRECISION,     -- median
    p95_value       DOUBLE PRECISION,     -- warning threshold source
    p99_value       DOUBLE PRECISION,     -- critical threshold source
    sample_count    INTEGER DEFAULT 0,
    last_computed   TIMESTAMPTZ DEFAULT NOW(),
    manual_override BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (server_id, metric_name)
);

-- ─── 5. Continuous Aggregates ────────────────────────────────
-- Pre-computed rollups for fast dashboard queries.

-- Hourly rollup
CREATE MATERIALIZED VIEW IF NOT EXISTS telemetry_hourly
WITH (timescaledb.continuous) AS
    SELECT
        time_bucket('1 hour', time) AS bucket,
        server_id,
        AVG(cpu_temp)       AS avg_cpu_temp,
        MAX(cpu_temp)       AS max_cpu_temp,
        MIN(cpu_temp)       AS min_cpu_temp,
        AVG(cpu_load)       AS avg_cpu_load,
        AVG(gpu_temp)       AS avg_gpu_temp,
        MAX(gpu_temp)       AS max_gpu_temp,
        AVG(gpu_power)      AS avg_gpu_power,
        AVG(ram_used_gb)    AS avg_ram_used,
        AVG(health_score)   AS avg_health,
        MIN(health_score)   AS min_health,
        COUNT(*)            AS sample_count
    FROM telemetry_raw
    GROUP BY bucket, server_id
WITH NO DATA;

-- Auto-refresh hourly aggregates every 30 minutes
SELECT add_continuous_aggregate_policy('telemetry_hourly',
    start_offset    => INTERVAL '3 hours',
    end_offset      => INTERVAL '1 hour',
    schedule_interval => INTERVAL '30 minutes',
    if_not_exists   => TRUE
);

-- Daily rollup
CREATE MATERIALIZED VIEW IF NOT EXISTS telemetry_daily
WITH (timescaledb.continuous) AS
    SELECT
        time_bucket('1 day', time) AS bucket,
        server_id,
        AVG(cpu_temp)       AS avg_cpu_temp,
        MAX(cpu_temp)       AS max_cpu_temp,
        AVG(gpu_temp)       AS avg_gpu_temp,
        MAX(gpu_temp)       AS max_gpu_temp,
        AVG(ram_used_gb)    AS avg_ram_used,
        AVG(health_score)   AS avg_health,
        MIN(health_score)   AS min_health,
        COUNT(*)            AS sample_count
    FROM telemetry_raw
    GROUP BY bucket, server_id
WITH NO DATA;

-- Auto-refresh daily aggregates every 6 hours
SELECT add_continuous_aggregate_policy('telemetry_daily',
    start_offset    => INTERVAL '3 days',
    end_offset      => INTERVAL '1 day',
    schedule_interval => INTERVAL '6 hours',
    if_not_exists   => TRUE
);

-- Prediction hourly summary
CREATE MATERIALIZED VIEW IF NOT EXISTS predictions_hourly
WITH (timescaledb.continuous) AS
    SELECT
        time_bucket('1 hour', time) AS bucket,
        server_id,
        model_name,
        COUNT(*)                    AS prediction_count,
        AVG(confidence)             AS avg_confidence,
        MIN(confidence)             AS min_confidence,
        AVG(ttf_hours)              AS avg_ttf_hours,
        AVG(anomaly_score)          AS avg_anomaly_score,
        AVG(inference_ms)           AS avg_inference_ms
    FROM predictions
    GROUP BY bucket, server_id, model_name
WITH NO DATA;

SELECT add_continuous_aggregate_policy('predictions_hourly',
    start_offset    => INTERVAL '3 hours',
    end_offset      => INTERVAL '1 hour',
    schedule_interval => INTERVAL '30 minutes',
    if_not_exists   => TRUE
);

-- ─── 6. Retention Policies ──────────────────────────────────
-- Raw data: 30 days. Aggregates: indefinite.
SELECT add_retention_policy('telemetry_raw', INTERVAL '30 days', if_not_exists => TRUE);
SELECT add_retention_policy('predictions', INTERVAL '90 days', if_not_exists => TRUE);

-- ─── 7. Compression (Optional, for production) ──────────────
-- Enable compression on telemetry_raw after 7 days
ALTER TABLE telemetry_raw SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'server_id',
    timescaledb.compress_orderby = 'time DESC'
);

SELECT add_compression_policy('telemetry_raw', INTERVAL '7 days', if_not_exists => TRUE);

-- ============================================================
--  Schema complete. Tables: servers, telemetry_raw, predictions, thresholds
--  Views: telemetry_hourly, telemetry_daily, predictions_hourly
--  Policies: 30d retention (raw), 90d retention (predictions),
--            7d compression, auto-refresh aggregates
-- ============================================================
