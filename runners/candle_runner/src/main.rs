use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig, DTYPE};
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use shared::*;
use std::time::Instant;
use tokenizers::{PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer};

fn main() -> Result<()> {
    let args = CommonArgs::parse();
    let corpus = Corpus::load(&args.corpus)?;
    let sentences = corpus.sentences(&args.length)?;
    if sentences.is_empty() {
        anyhow::bail!("corpus bucket {} is empty", args.length);
    }

    let device = Device::Cpu;
    let mut mem = MemProbe::new();
    let _ = mem.sample_mb();

    let cold = Instant::now();
    let api = Api::new()?;
    let repo = api.repo(Repo::new(
        "sentence-transformers/all-MiniLM-L6-v2".into(),
        RepoType::Model,
    ));
    let config_path = repo.get("config.json")?;
    let tokenizer_path = repo.get("tokenizer.json")?;
    let weights_path = repo
        .get("model.safetensors")
        .or_else(|_| repo.get("pytorch_model.bin"))?;

    // nosemgrep: rust.actix.path-traversal.tainted-path.tainted-path
    let config_file = std::fs::File::open(&config_path)?;
    let config: BertConfig = serde_json::from_reader(config_file)?;
    let mut tokenizer =
        Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow!("tokenizer: {e}"))?;
    tokenizer.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::BatchLongest,
        direction: PaddingDirection::Right,
        pad_to_multiple_of: None,
        pad_id: 0,
        pad_type_id: 0,
        pad_token: "[PAD]".to_string(),
    }));
    tokenizer
        .with_truncation(None)
        .map_err(|e| anyhow!("tokenizer: {e}"))?;
    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DTYPE, &device)? };
    let model = BertModel::load(vb, &config)?;
    let cold_start_ms = elapsed_ms(cold);

    let embed_batch = |inputs: &[&str]| -> Result<Vec<Vec<f32>>> {
        let encs = tokenizer
            .encode_batch(inputs.to_vec(), true)
            .map_err(|e| anyhow!("encode: {e}"))?;
        let b = encs.len();
        let l = encs.first().map(|e| e.get_ids().len()).unwrap_or(0);

        let mut ids_flat: Vec<u32> = Vec::with_capacity(b * l);
        let mut mask_flat: Vec<u32> = Vec::with_capacity(b * l);
        for enc in &encs {
            ids_flat.extend_from_slice(enc.get_ids());
            mask_flat.extend_from_slice(enc.get_attention_mask());
        }

        let token_ids = Tensor::from_vec(ids_flat, (b, l), &device)?;
        let attn = Tensor::from_vec(mask_flat, (b, l), &device)?;
        let type_ids = token_ids.zeros_like()?;
        let hidden = model.forward(&token_ids, &type_ids, Some(&attn))?;

        let mask_f = attn.to_dtype(DType::F32)?.unsqueeze(2)?;
        let masked = hidden.broadcast_mul(&mask_f)?;
        let summed = masked.sum(1)?;
        let counts = mask_f.sum(1)?;
        let pooled = summed.broadcast_div(&counts)?;
        let norm = pooled.sqr()?.sum_keepdim(1)?.sqrt()?;
        let normalized = pooled.broadcast_div(&norm)?;

        let mut results = Vec::with_capacity(b);
        for bi in 0..b {
            let row: Vec<f32> = normalized.get(bi)?.to_vec1()?;
            results.push(row);
        }
        Ok(results)
    };

    let _ = embed_batch(&[sentences[0].as_str()])?;
    let rss_idle_mb = mem.sample_mb();

    for i in 0..args.warmup {
        embed_batch(&[sentences[i % sentences.len()].as_str()])?;
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
        let out = embed_batch(&batch)?;
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
        let all = embed_batch(&refs)?;
        save_vectors_bin(path, &all)?;
    }

    let result = BenchResult {
        backend: "candle".into(),
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
