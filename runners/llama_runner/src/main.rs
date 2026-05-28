use anyhow::{anyhow, Result};
use clap::Parser;
use llama_cpp_2::{
    context::{
        params::{LlamaContextParams, LlamaPoolingType},
        LlamaContext,
    },
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel},
    token::LlamaToken,
};
use shared::*;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
struct LlamaArgs {
    #[command(flatten)]
    common: CommonArgs,

    /// Path to the GGUF model file. The all-MiniLM-L6-v2 F16 GGUF that ollama uses
    /// lives at ~/.ollama/models/blobs/<sha>; the run_all.sh script copies it into
    /// models/all-MiniLM-L6-v2-gguf/model.gguf.
    #[arg(long, default_value = "models/all-MiniLM-L6-v2-gguf/model.gguf")]
    model: PathBuf,
}

fn main() -> Result<()> {
    let args = LlamaArgs::parse();
    let common = &args.common;
    let corpus = Corpus::load(&common.corpus)?;
    let sentences = corpus.sentences(&common.length)?;
    if sentences.is_empty() {
        anyhow::bail!("corpus bucket {} is empty", common.length);
    }

    let mut mem = MemProbe::new();
    let _ = mem.sample_mb();

    let cold = Instant::now();
    let backend = LlamaBackend::init().map_err(|e| anyhow!("backend: {e}"))?;
    let model_params = LlamaModelParams::default();
    // nosemgrep: rust.actix.path-traversal.tainted-path.tainted-path
    let model = LlamaModel::load_from_file(&backend, &args.model, &model_params)
        .map_err(|e| anyhow!("model load: {e}"))?;

    // Context sized to hold the full batch worth of tokens at the longest sequence.
    let n_ctx: u32 = 512;
    let n_batch_tokens: u32 = (common.batch as u32 * n_ctx).max(n_ctx);
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx))
        .with_n_batch(n_batch_tokens)
        .with_n_ubatch(n_batch_tokens)
        .with_n_seq_max(64)
        .with_n_threads(common.threads as i32)
        .with_n_threads_batch(common.threads as i32)
        .with_embeddings(true)
        .with_pooling_type(LlamaPoolingType::Mean);
    let mut ctx = model
        .new_context(&backend, ctx_params)
        .map_err(|e| anyhow!("ctx: {e}"))?;
    let cold_start_ms = elapsed_ms(cold);

    let embed_batch = |inputs: &[&str], ctx: &mut LlamaContext| -> Result<Vec<Vec<f32>>> {
        let mut all_tokens: Vec<Vec<LlamaToken>> = Vec::with_capacity(inputs.len());
        let mut total_tokens = 0usize;
        for s in inputs {
            let tokens = model
                .str_to_token(s, AddBos::Always)
                .map_err(|e| anyhow!("tokenize: {e}"))?;
            total_tokens += tokens.len();
            all_tokens.push(tokens);
        }
        let mut batch = LlamaBatch::new(total_tokens.max(1), inputs.len() as i32);
        for (seq_id, tokens) in all_tokens.iter().enumerate() {
            for (pos, &token) in tokens.iter().enumerate() {
                batch
                    .add(
                        token,
                        pos as i32,
                        &[seq_id as i32],
                        pos == tokens.len() - 1,
                    )
                    .map_err(|e| anyhow!("batch add: {e}"))?;
            }
        }
        ctx.clear_kv_cache();
        ctx.encode(&mut batch).map_err(|e| anyhow!("encode: {e}"))?;

        let mut results = Vec::with_capacity(inputs.len());
        for seq_id in 0..inputs.len() {
            let emb = ctx
                .embeddings_seq_ith(seq_id as i32)
                .map_err(|e| anyhow!("emb: {e}"))?;
            let norm = emb.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            let normalized: Vec<f32> = emb.iter().map(|x| x / norm).collect();
            results.push(normalized);
        }
        Ok(results)
    };

    let _ = embed_batch(&[sentences[0].as_str()], &mut ctx)?;
    let rss_idle_mb = mem.sample_mb();

    for i in 0..common.warmup {
        embed_batch(&[sentences[i % sentences.len()].as_str()], &mut ctx)?;
    }

    let mut latencies = Vec::with_capacity(common.measure);
    let throughput_start = Instant::now();
    let mut idx = 0usize;
    while latencies.len() < common.measure {
        let mut batch_in: Vec<&str> = Vec::with_capacity(common.batch);
        for _ in 0..common.batch {
            batch_in.push(sentences[idx % sentences.len()].as_str());
            idx += 1;
        }
        let t = Instant::now();
        let out = embed_batch(&batch_in, &mut ctx)?;
        let per_item = elapsed_ms(t) / out.len().max(1) as f64;
        for _ in 0..out.len() {
            latencies.push(per_item);
            if latencies.len() >= common.measure {
                break;
            }
        }
        let _ = mem.sample_mb();
    }
    let total_ms = elapsed_ms(throughput_start);
    let throughput_eps = (common.measure as f64) / (total_ms / 1000.0);

    if let Some(path) = &common.save_vectors {
        let refs: Vec<&str> = sentences.iter().map(String::as_str).collect();
        let all = embed_batch(&refs, &mut ctx)?;
        save_vectors_bin(path, &all)?;
    }

    let result = BenchResult {
        backend: "llama".into(),
        config: build_run_config(common),
        metrics: Metrics {
            latency_ms: Latency::from_samples(&latencies),
            throughput_eps,
            cold_start_ms,
            rss_idle_mb,
            rss_peak_mb: mem.peak_mb(),
        },
        vectors_path: common.save_vectors.as_ref().map(|p| p.display().to_string()),
        raw_latencies_ms: latencies,
    };
    write_result(&common.out, &result)?;
    Ok(())
}
