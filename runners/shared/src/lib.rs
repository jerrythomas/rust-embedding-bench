use anyhow::{Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;
use sysinfo::{Pid, System};

#[derive(Parser, Debug, Clone)]
pub struct CommonArgs {
    #[arg(long, default_value = "corpus/sentences.json")]
    pub corpus: PathBuf,

    #[arg(long, default_value = "short")]
    pub length: String,

    #[arg(long, default_value_t = 1)]
    pub batch: usize,

    #[arg(long, default_value_t = 1)]
    pub threads: usize,

    #[arg(long, default_value_t = 50)]
    pub warmup: usize,

    #[arg(long, default_value_t = 500)]
    pub measure: usize,

    #[arg(long, default_value = "result.json")]
    pub out: PathBuf,

    #[arg(long)]
    pub save_vectors: Option<PathBuf>,

    #[arg(long, default_value = "fp32")]
    pub precision: String,
}

#[derive(Deserialize)]
pub struct Corpus {
    pub buckets: HashMap<String, Vec<String>>,
}

impl Corpus {
    pub fn load(path: &Path) -> Result<Self> {
        // Operator-supplied CLI arg for a local benchmark tool, not network input.
        // nosemgrep: rust.actix.path-traversal.tainted-path.tainted-path
        let data = std::fs::read_to_string(path)
            .with_context(|| format!("read corpus {:?}", path))?;
        Ok(serde_json::from_str(&data)?)
    }

    pub fn sentences(&self, bucket: &str) -> Result<&[String]> {
        self.buckets
            .get(bucket)
            .map(|v| v.as_slice())
            .with_context(|| format!("bucket not in corpus: {}", bucket))
    }
}

#[derive(Serialize)]
pub struct BenchResult {
    pub backend: String,
    pub config: RunConfig,
    pub metrics: Metrics,
    pub vectors_path: Option<String>,
    pub raw_latencies_ms: Vec<f64>,
}

#[derive(Serialize)]
pub struct RunConfig {
    pub batch_size: usize,
    pub threads: usize,
    pub length: String,
    pub precision: String,
    pub warmup: usize,
    pub measure: usize,
}

#[derive(Serialize)]
pub struct Metrics {
    pub latency_ms: Latency,
    pub throughput_eps: f64,
    pub cold_start_ms: f64,
    pub rss_idle_mb: f64,
    pub rss_peak_mb: f64,
}

#[derive(Serialize, Default)]
pub struct Latency {
    pub mean: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
}

impl Latency {
    pub fn from_samples(samples: &[f64]) -> Self {
        if samples.is_empty() {
            return Self::default();
        }
        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let pct = |p: f64| -> f64 {
            let idx = (((sorted.len() - 1) as f64) * p).round() as usize;
            sorted[idx.min(sorted.len() - 1)]
        };
        Self {
            mean: samples.iter().sum::<f64>() / samples.len() as f64,
            p50: pct(0.50),
            p95: pct(0.95),
            p99: pct(0.99),
        }
    }
}

pub struct MemProbe {
    sys: System,
    pid: Pid,
    peak_bytes: u64,
}

impl MemProbe {
    pub fn new() -> Self {
        Self {
            sys: System::new(),
            pid: Pid::from(std::process::id() as usize),
            peak_bytes: 0,
        }
    }

    pub fn sample_mb(&mut self) -> f64 {
        self.sys.refresh_process(self.pid);
        let bytes = self.sys.process(self.pid).map(|p| p.memory()).unwrap_or(0);
        if bytes > self.peak_bytes {
            self.peak_bytes = bytes;
        }
        bytes as f64 / 1024.0 / 1024.0
    }

    pub fn peak_mb(&self) -> f64 {
        self.peak_bytes as f64 / 1024.0 / 1024.0
    }
}

impl Default for MemProbe {
    fn default() -> Self {
        Self::new()
    }
}

pub fn save_vectors_bin(path: &Path, vectors: &[Vec<f32>]) -> Result<()> {
    use std::io::Write;
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).ok();
        }
    }
    // Operator-supplied CLI arg for a local benchmark tool, not network input.
    // nosemgrep: rust.actix.path-traversal.tainted-path.tainted-path
    let mut f = std::fs::File::create(path)?;
    for v in vectors {
        for &x in v {
            f.write_all(&x.to_le_bytes())?;
        }
    }
    Ok(())
}

pub fn write_result(path: &Path, result: &BenchResult) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).ok();
        }
    }
    // Operator-supplied CLI arg for a local benchmark tool, not network input.
    // nosemgrep: rust.actix.path-traversal.tainted-path.tainted-path
    std::fs::write(path, serde_json::to_string_pretty(result)?)?;
    Ok(())
}

pub fn elapsed_ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

pub fn build_run_config(common: &CommonArgs) -> RunConfig {
    RunConfig {
        batch_size: common.batch,
        threads: common.threads,
        length: common.length.clone(),
        precision: common.precision.clone(),
        warmup: common.warmup,
        measure: common.measure,
    }
}
