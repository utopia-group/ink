use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use tracing_subscriber::EnvFilter;

mod latest;
mod trivial;
use rink::lang::json::ExprJson;
use rink::lang::{Expr, Type};
use rink::syn::*;

#[derive(Deserialize, Serialize)]
struct BenchmarkJson {
    acc: ExprJson,
    init: ExprJson,
}

#[macro_export]
macro_rules! benchmarks {
    ($($x:ident),*) => {
        HashMap::from([
            $((name_of!($x), $x())),*
        ])
    };
}

#[derive(Clone, Debug)]
struct Benchmark {
    init: Expr,
    accumulator: Expr,
    is_homomorphic: Option<bool>,
}

#[derive(Serialize, Deserialize)]
struct NormalizerSynthesisResult {
    status: String,
    normalizer: Result<String, String>,
    input_stats: Stats,
    input_ast_size: usize,
    normalizer_ast_size: usize,
    elapsed: f32,
}

#[derive(Serialize, Deserialize)]
struct Stats {
    has_tuple: bool,
    has_collection: bool,
    has_nested_collection: bool,
    has_conditional: bool,
}

#[derive(ValueEnum, Debug, Clone)]
enum RunMode {
    /// Run Ink
    RunInk,

    /// Run CVC5 baseline
    RunCvc5,

    /// Run ablation NoReduction; use the merge operator spec + top-level refutation
    RunNoReduction,

    /// Run ablation NoDecompose
    RunNoDecompose,

    /// Run ablation NoDeductive; only use the Norm-Synth rule
    RunNoDeductive,

    /// Run ablation NoRefute; refutation rules are disabled
    RunNoRefute,
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
#[command(propagate_version = true)]
struct Args {
    #[clap(value_enum, default_value_t=RunMode::RunInk, long, short)]
    mode: RunMode,

    /// Load benchmark from JSON files. If specified, this should be the basename (e.g., "example" for "example.json" and "example.init.json")
    #[clap(long)]
    json: Option<String>,

    /// Test suite name (used when not loading from JSON)
    suite: Option<String>,

    /// Benchmark name (used when not loading from JSON)
    benchmark: Option<String>,
}

fn analyze_program(expr: &Expr, acc_type: &Type) -> Stats {
    let has_tuple = acc_type.into_iter().any(|t| t.is_compound());
    let has_collection = acc_type.into_iter().any(|t| t.is_collection());
    let has_nested_collection = acc_type
        .into_iter()
        .any(|t| t.is_collection() && t.element_type().into_iter().any(|t| t.is_collection()));
    let has_conditional = expr.into_iter().any(|e| matches!(e, Expr::Ite { .. }));
    Stats {
        has_tuple,
        has_collection,
        has_nested_collection,
        has_conditional,
    }
}

fn load_benchmark_from_json(json_path: &str) -> Result<Benchmark, Box<dyn std::error::Error>> {
    let json_content = fs::read_to_string(&json_path)
        .map_err(|e| format!("Failed to read {}: {}", json_path, e))?;

    let benchmark_json: BenchmarkJson = serde_json::from_str(&json_content)
        .map_err(|e| format!("Failed to parse {}: {}", json_path, e))?;

    // For JSON-loaded benchmarks, we don't assume whether they are homomorphic
    // The user can specify this separately if needed
    Ok(Benchmark {
        init: benchmark_json.init.into(),
        accumulator: benchmark_json.acc.into(),
        is_homomorphic: None,
    })
}

fn load_benchmark(args: &Args) -> Result<Benchmark, Box<dyn std::error::Error>> {
    if let Some(ref json_path) = args.json {
        load_benchmark_from_json(json_path)
    } else if let (Some(ref suite), Some(ref benchmark)) = (&args.suite, &args.benchmark) {
        let benchmarks = match suite.as_str() {
            "latest" => latest::benchmarks(),
            "trivial" => trivial::benchmarks(),
            _ => return Err("unknown test suite".into()),
        };

        benchmarks
            .get(benchmark.as_str())
            .map(|b| (*b).clone())
            .ok_or_else(|| {
                format!("benchmark '{}' not found in suite '{}'", benchmark, suite).into()
            })
    } else {
        Err("Either --json or both suite and benchmark must be provided".into())
    }
}

fn run_benchmark(benchmark: Benchmark, features: SynthesisFeatures) -> NormalizerSynthesisResult {
    let Benchmark {
        init,
        accumulator,
        is_homomorphic,
    } = &benchmark;
    eprintln!("============ initializer ============\n{}\n", init);
    eprintln!("============ accumulator ============\n{}\n", accumulator);

    let time_start = std::time::Instant::now();
    let result = check_homomorphism(accumulator, init, features);
    let elapsed = time_start.elapsed().as_secs_f32();

    let input_ast_size = accumulator.ast_size() + init.ast_size();
    let normalizer_ast_size = result.as_ref().map(|n| n.ast_size()).unwrap_or(0);
    let input_stats = analyze_program(accumulator, {
        let (_, params) = accumulator.uncurry_lambda();
        params.first().unwrap().1
    });

    match (result, is_homomorphic) {
        (Ok(_), Some(false)) | (Err(NormalizerSynthesisFailure::Refuted), Some(true)) => {
            NormalizerSynthesisResult {
                status: "Failure".to_string(),
                normalizer: Err("Result not consistent to the ground truth".to_string()),
                input_stats,
                input_ast_size,
                normalizer_ast_size,
                elapsed,
            }
        }
        (Ok(norm), _) => NormalizerSynthesisResult {
            status: "Success".to_string(),
            normalizer: Ok(norm.to_string()),
            input_stats,
            input_ast_size,
            normalizer_ast_size,
            elapsed,
        },
        (Err(NormalizerSynthesisFailure::Refuted), _) => NormalizerSynthesisResult {
            status: "Refuted".to_string(),
            normalizer: Ok("".to_string()),
            input_stats,
            input_ast_size,
            normalizer_ast_size,
            elapsed,
        },
        (Err(NormalizerSynthesisFailure::Timeout), _) => NormalizerSynthesisResult {
            status: "Timeout".to_string(),
            normalizer: Ok("".to_string()),
            input_stats,
            input_ast_size,
            normalizer_ast_size,
            elapsed,
        },
        (Err(error), _) => NormalizerSynthesisResult {
            status: "Error".to_string(),
            normalizer: Err(format!("{:?}", error)),
            input_stats,
            input_ast_size,
            normalizer_ast_size,
            elapsed,
        },
    }
}

fn main() {
    tracing_subscriber::fmt()
        .with_writer(io::stderr)
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let args = Args::parse();
    let benchmark = load_benchmark(&args).expect("failed to load benchmark");
    let features: SynthesisFeatures = match args.mode {
        RunMode::RunInk => Default::default(),
        RunMode::RunCvc5 => SynthesisFeatures::None,
        RunMode::RunNoReduction => {
            SynthesisFeatures::default().difference(SynthesisFeatures::Reduction)
        }
        RunMode::RunNoDecompose => {
            SynthesisFeatures::default().difference(SynthesisFeatures::Decomposition)
        }
        RunMode::RunNoDeductive => SynthesisFeatures::default()
            .difference(SynthesisFeatures::Deduction | SynthesisFeatures::Decomposition),
        RunMode::RunNoRefute => {
            SynthesisFeatures::default().difference(SynthesisFeatures::Refutation)
        }
    };
    let result = run_benchmark(benchmark, features);

    serde_json::to_writer(io::stdout(), &result).expect("failed to write result");
    println!()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn export_benchmark_to_json(benchmark: &Benchmark) -> String {
        let Benchmark {
            init, accumulator, ..
        } = benchmark;
        let benchmark_json = BenchmarkJson {
            acc: accumulator.into(),
            init: init.into(),
        };

        serde_json::to_string_pretty(&benchmark_json)
            .expect("failed to serialize benchmark to JSON")
    }

    fn export_latest(export_dir: &str) {
        let benchmarks = latest::benchmarks();
        for (name, benchmark) in benchmarks {
            let json_content = export_benchmark_to_json(&benchmark);
            let file_path = format!("{}/{}.json", export_dir, name);
            fs::write(file_path, json_content).expect("failed to write benchmark JSON");
        }
    }

    #[test]
    fn test_export_latest() {
        let export_dir = "exported_benchmarks";
        fs::create_dir_all(export_dir).expect("failed to create export directory");
        export_latest(export_dir);
    }
}
