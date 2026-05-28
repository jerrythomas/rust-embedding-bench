use anyhow::Result;
use clap::Parser;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use shared::*;
use std::time::Instant;

fn main() -> Result<()> {
    let args = CommonArgs::parse();
    let corpus = Corpus::load(&args.corpus)?;
    let sentences = corpus.sentences(&args.length)?;
    if sentences.is_empty() {
        anyhow::bail!("corpus bucket {} is empty", args.length);
    }

    let mut mem = MemProbe::new();
    let _ = mem.sample_mb();

    let cold = Instant::now();
    let mut model = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::AllMiniLML6V2).with_show_download_progress(false),
    )?;
    let cold_start_ms = elapsed_ms(cold);

    let _ = model.embed(vec![&sentences[0]], Some(1))?;
    let rss_idle_mb = mem.sample_mb();

    for i in 0..args.warmup {
        let s = &sentences[i % sentences.len()];
        model.embed(vec![s], Some(args.batch))?;
    }

    let mut latencies = Vec::with_capacity(args.measure);
    let throughput_start = Instant::now();
    let mut idx = 0usize;
    while latencies.len() < args.measure {
        let mut batch: Vec<&str> = Vec::with_capacity(args.batch);
        for _ in 0..args.batch {
            batch.push(sentences[idx % sentences.len()].as_str());
            idx += 1;
        }
        let t = Instant::now();
        let out = model.embed(batch, Some(args.batch))?;
        let per_item = elapsed_ms(t) / out.len().max(1) as f64;
        for _ in 0..out.len() {
            latencies.push(per_item);
            if latencies.len() >= args.measure {
                break;
            }
        }
        let _ = mem.sample_mb();
    }
    let total_ms = elapsed_ms(throughput_start);
    let throughput_eps = (args.measure as f64) / (total_ms / 1000.0);

    if let Some(path) = &args.save_vectors {
        let refs: Vec<&str> = sentences.iter().map(String::as_str).collect();
        let all = model.embed(refs, Some(args.batch))?;
        save_vectors_bin(path, &all)?;
    }

    let result = BenchResult {
        backend: "fastembed".into(),
        config: build_run_config(&args),
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
