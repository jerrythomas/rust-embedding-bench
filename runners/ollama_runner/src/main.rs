use anyhow::{Context, Result};
use clap::Parser;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use shared::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Parser, Debug)]
struct OllamaArgs {
    #[command(flatten)]
    common: CommonArgs,

    /// Number of worker threads firing concurrent HTTP requests to the ollama daemon.
    /// 1 = sequential (default). >1 = parallel pipelined to amortize HTTP/JSON overhead.
    #[arg(long, default_value_t = 1)]
    concurrency: usize,
}

#[derive(Serialize)]
struct EmbedReq<'a> {
    model: &'a str,
    input: Vec<&'a str>,
    keep_alive: i64,
}

#[derive(Deserialize)]
struct EmbedResp {
    embeddings: Vec<Vec<f32>>,
}

fn main() -> Result<()> {
    let parsed = OllamaArgs::parse();
    let args = &parsed.common;
    let concurrency = parsed.concurrency.max(1);
    let corpus = Corpus::load(&args.corpus)?;
    let sentences = corpus.sentences(&args.length)?;
    if sentences.is_empty() {
        anyhow::bail!("corpus bucket {} is empty", args.length);
    }

    let host = std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".into());
    let url = format!("{}/api/embed", host.trim_end_matches('/'));
    let model_name = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "all-minilm".into());

    let client = Client::builder().timeout(Duration::from_secs(120)).build()?;

    let mut mem = MemProbe::new();
    let _ = mem.sample_mb();

    let cold = Instant::now();
    let _ = embed(&client, &url, &model_name, &[sentences[0].as_str()])
        .context("initial embed (model load + first request)")?;
    let cold_start_ms = elapsed_ms(cold);
    let rss_idle_mb = mem.sample_mb();

    for i in 0..args.warmup {
        let s = sentences[i % sentences.len()].as_str();
        let _ = embed(&client, &url, &model_name, &[s])?;
    }

    let latencies = Arc::new(Mutex::new(Vec::with_capacity(args.measure)));
    let counter = Arc::new(AtomicUsize::new(0));
    let target = args.measure;
    let batch_sz = args.batch;
    let client = Arc::new(client);
    let url = Arc::new(url.clone());
    let model_name = Arc::new(model_name.clone());
    let sentences_arc: Arc<Vec<String>> = Arc::new(sentences.to_vec());

    let throughput_start = Instant::now();
    let mut handles = Vec::with_capacity(concurrency);
    for wid in 0..concurrency {
        let counter = counter.clone();
        let client = client.clone();
        let url = url.clone();
        let model_name = model_name.clone();
        let sentences = sentences_arc.clone();
        let latencies = latencies.clone();
        handles.push(thread::spawn(move || -> Result<()> {
            let mut idx = wid * batch_sz;
            loop {
                let claimed = counter.fetch_add(batch_sz, Ordering::Relaxed);
                if claimed >= target {
                    break;
                }
                let batch: Vec<&str> = (0..batch_sz)
                    .map(|_| {
                        let s = sentences[idx % sentences.len()].as_str();
                        idx += 1;
                        s
                    })
                    .collect();
                let t = Instant::now();
                let out = embed(&client, url.as_str(), model_name.as_str(), &batch)?;
                let per_item = elapsed_ms(t) / out.len().max(1) as f64;
                let mut lats = latencies.lock().unwrap();
                for _ in 0..out.len() {
                    lats.push(per_item);
                }
            }
            Ok(())
        }));
    }
    for h in handles {
        h.join().map_err(|_| anyhow::anyhow!("worker panic"))??;
    }
    let total_ms = elapsed_ms(throughput_start);
    let _ = mem.sample_mb();
    let latencies = Arc::try_unwrap(latencies)
        .map_err(|_| anyhow::anyhow!("dangling latencies arc"))?
        .into_inner()?;
    let measured = latencies.len().min(target);
    let throughput_eps = (measured as f64) / (total_ms / 1000.0);

    if let Some(path) = &args.save_vectors {
        let refs: Vec<&str> = sentences.iter().map(String::as_str).collect();
        let all = embed(&client, url.as_str(), model_name.as_str(), &refs)?;
        save_vectors_bin(path, &all)?;
    }

    let result = BenchResult {
        backend: "ollama".into(),
        config: build_run_config(args),
        metrics: Metrics {
            latency_ms: Latency::from_samples(&latencies),
            throughput_eps,
            cold_start_ms,
            rss_idle_mb,
            rss_peak_mb: mem.peak_mb(),
        },
        vectors_path: args.save_vectors.as_ref().map(|p| p.display().to_string()),
        raw_latencies_ms: latencies,
    };
    write_result(&args.out, &result)?;
    Ok(())
}

fn embed(client: &Client, url: &str, model: &str, inputs: &[&str]) -> Result<Vec<Vec<f32>>> {
    let req = EmbedReq {
        model,
        input: inputs.to_vec(),
        keep_alive: -1,
    };
    let resp: EmbedResp = client
        .post(url)
        .json(&req)
        .send()?
        .error_for_status()?
        .json()?;
    Ok(resp.embeddings)
}
