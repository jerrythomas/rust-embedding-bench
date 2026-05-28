use anyhow::{anyhow, Result};
use clap::Parser;
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};
use shared::*;
use std::path::PathBuf;
use std::time::Instant;
use tokenizers::{PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer};

#[derive(Parser, Debug)]
struct OrtArgs {
    #[command(flatten)]
    common: CommonArgs,

    /// Path to ONNX model. Export with:
    ///   optimum-cli export onnx --model sentence-transformers/all-MiniLM-L6-v2 models/all-MiniLM-L6-v2
    #[arg(long, default_value = "models/all-MiniLM-L6-v2/model.onnx")]
    model: PathBuf,

    #[arg(long, default_value = "models/all-MiniLM-L6-v2/tokenizer.json")]
    tokenizer: PathBuf,

    /// Backend label written to the result JSON (use "ort-int8" for the quantized model).
    #[arg(long, default_value = "ort")]
    backend_label: String,
}

fn oe<E: std::fmt::Display>(e: E) -> anyhow::Error {
    anyhow!("ort: {e}")
}

fn main() -> Result<()> {
    let args = OrtArgs::parse();
    let common = &args.common;
    let corpus = Corpus::load(&common.corpus)?;
    let sentences = corpus.sentences(&common.length)?;
    if sentences.is_empty() {
        anyhow::bail!("corpus bucket {} is empty", common.length);
    }

    let mut mem = MemProbe::new();
    let _ = mem.sample_mb();

    let cold = Instant::now();
    let mut session = Session::builder()
        .map_err(oe)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(oe)?
        .with_intra_threads(common.threads)
        .map_err(oe)?
        // nosemgrep: rust.actix.path-traversal.tainted-path.tainted-path
        .commit_from_file(&args.model)
        .map_err(oe)?;
    // nosemgrep: rust.actix.path-traversal.tainted-path.tainted-path
    let mut tokenizer =
        Tokenizer::from_file(&args.tokenizer).map_err(|e| anyhow!("tokenizer: {e}"))?;
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
    let cold_start_ms = elapsed_ms(cold);

    let embed_batch = |inputs: &[&str], sess: &mut Session| -> Result<Vec<Vec<f32>>> {
        let encs = tokenizer
            .encode_batch(inputs.to_vec(), true)
            .map_err(|e| anyhow!("encode: {e}"))?;
        let b = encs.len();
        let l = encs.first().map(|e| e.get_ids().len()).unwrap_or(0);

        let mut ids: Vec<i64> = Vec::with_capacity(b * l);
        let mut mask: Vec<i64> = Vec::with_capacity(b * l);
        for enc in &encs {
            ids.extend(enc.get_ids().iter().map(|&x| x as i64));
            mask.extend(enc.get_attention_mask().iter().map(|&x| x as i64));
        }
        let tt: Vec<i64> = vec![0; b * l];

        let ids_t = Tensor::<i64>::from_array(([b, l], ids)).map_err(oe)?;
        let mask_t = Tensor::<i64>::from_array(([b, l], mask.clone())).map_err(oe)?;
        let tt_t = Tensor::<i64>::from_array(([b, l], tt)).map_err(oe)?;

        let outputs = sess
            .run(ort::inputs![
                "input_ids" => ids_t,
                "attention_mask" => mask_t,
                "token_type_ids" => tt_t,
            ])
            .map_err(oe)?;

        let (shape, data) = outputs[0].try_extract_tensor::<f32>().map_err(oe)?;
        if shape.len() < 3 {
            anyhow::bail!("expected rank-3 output, got shape {:?}", &shape[..]);
        }
        let l_out = shape[shape.len() - 2] as usize;
        let h = shape[shape.len() - 1] as usize;

        let mut results = Vec::with_capacity(b);
        for bi in 0..b {
            let mut pooled = vec![0f32; h];
            let mut count = 0f32;
            for ti in 0..l_out {
                if mask[bi * l + ti] == 1 {
                    let off = (bi * l_out + ti) * h;
                    for d in 0..h {
                        pooled[d] += data[off + d];
                    }
                    count += 1.0;
                }
            }
            if count > 0.0 {
                for v in pooled.iter_mut() {
                    *v /= count;
                }
            }
            let norm = pooled.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            for v in pooled.iter_mut() {
                *v /= norm;
            }
            results.push(pooled);
        }
        Ok(results)
    };

    let _ = embed_batch(&[sentences[0].as_str()], &mut session)?;
    let rss_idle_mb = mem.sample_mb();

    for i in 0..common.warmup {
        let s = sentences[i % sentences.len()].as_str();
        embed_batch(&[s], &mut session)?;
    }

    let mut latencies = Vec::with_capacity(common.measure);
    let throughput_start = Instant::now();
    let mut idx = 0usize;
    while latencies.len() < common.measure {
        let mut batch: Vec<&str> = Vec::with_capacity(common.batch);
        for _ in 0..common.batch {
            batch.push(sentences[idx % sentences.len()].as_str());
            idx += 1;
        }
        let t = Instant::now();
        let out = embed_batch(&batch, &mut session)?;
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
        let all = embed_batch(&refs, &mut session)?;
        save_vectors_bin(path, &all)?;
    }

    let result = BenchResult {
        backend: args.backend_label.clone(),
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
