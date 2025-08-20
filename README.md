# Ink

Ink is a synthesis tool for merge operators for User-Defined Aggregate Functions (UDAFs), written in Rust and using Nix for dependency management. A recent installation of Nix (version 2.18.1 or higher) is the only prerequisite to get started. Additionally, we offer a Docker-based solution for running Nix.

We support both x86_64 and aarch64 platforms. Note that the binary cache for prebuilt dependencies is only available for the x86_64 architecture. On an aarch64 system, building the dependencies might require an additional 10-15 minutes.

The full evaluation of Ink requires CVC5 and ParSynt. Please follow the instructions below to configure your environment.

## Documentation

Below, we provide an overview of the Ink project's directory structure:

- `src/` contains all the sources related to Ink.
    * `eval/` contains the main evaluation executable that showcases the use of Ink. It loads benchmarks and finds merge functions for them.
        - `main.rs` is the entrypoint that handles command line arguments and orchestrates the synthesis process.
    * `incr/` implements expression decomposition techniques (Figure 13, 14 in the paper).
        - `simp/` contains function simplification rules (Figure 14).
        - `vd/` implements UDAF decomposition rules (Figure 13).
        - `incr.rs` defines data structures used in simplification and decomposition.
    * `syn/` implements normalizer validation and refutation (Figure 11 in the paper).
        - `assumption.rs` handles additional assumptions used in SyGuS encoding.
        - `refute.rs` provides fuzzing-based refutation implementation.
        - `sygus.rs` implements the SyGuS language encoding procedure.
        - `syn.rs` contains the top-level normalizer validation and refutation implementation.
    * `lang/` defines the DSL including expressions and typing system, interpreter, type inference, and egg-based program rewriter/optimizer.
    * `lib.rs` is the main library entry point.

- `evaluation/` contains scripts and benchmarks for evaluation.
    * `evaluation.sh` is the main evaluation script that runs all experiments.
    * `eval.py` analyzes results and generates figures.
    * `benchmarks/` contains JSON benchmark files for User-Defined Aggregate Functions (UDAFs).
    * Shell scripts for individual tool evaluations (`ink.sh`, `cvc5.sh`, `parsynt.sh`, etc.).

- `transpile/` contains Python utilities for transpilation tasks.

## Usage

First, enter the development environment:

```shell
$ nix develop
```

> [!NOTE]  
> You can safely ignore some warning message about z3 not installed (`Rosette installed successfully, but wasn't able to install the Z3 SMT solver.`). This is actually not an error:
> Since we use a Nix container, Nix handles all dependency management including Z3 installation. Rosette doesn't need to install Z3 itself because we manually provide it through our Nix configuration.
> If you're curious about the implementation details, you can inspect line 111 of flake.nix where you'll see:
>
> ```ln -s ${pkgs.z3}/bin/z3 ~/.local/share/racket/8.14/pkgs/rosette/bin/z3 || true```
>
> This line creates a symbolic link to place the Nix-managed Z3 binary at exactly the path that Rosette expects (`/root/.local/share/racket/8.14/pkgs/rosette/bin/z3`), which is the same location mentioned in the error message.

Once in the development environment, you can use the main evaluation executable:

```shell
$ rink-eval --help
```

This will show available command-line options for the synthesis tool.

To run Ink on a specific benchmark file:

```shell
$ rink-eval --json evaluation/benchmarks/counter.json
```

## Getting Started with Nix

Prerequisites:
* Install the [Nix package manager](https://zero-to-nix.com/start/install/).
* If you didn't use the provided link above to install Nix, make sure to enable [Nix flakes](https://nixos.wiki/wiki/Flakes).

To begin, execute the following commands:

```shell
$ nix develop
```

You can move forward to the section *Step-by-Step Instructions of Ink Evaluation*.

## Getting Started with Nix Container

To start, build the Docker image with the following command:
   
```shell
$ docker build . -t ink_ae:latest
```

Then, create a container from the built image, mapping the current project directory to a path within the container:

```shell
$ docker run --rm -v $(pwd):/workspace -it ink_ae:latest
```

You can move forward to the section *Step-by-Step Instructions of Ink Evaluation*.

## Step-by-Step Instructions of Ink Evaluation

0. Enter the development environment for Ink

```shell
$ nix develop
```

1. Verify the environment

Verify the environment by testing Ink on a simple example:

```shell
$ rink-eval --json evaluation/benchmarks/counter.json
# This should successfully synthesize a merge function for the counter benchmark
```

2. Run the evaluation script

We have provided a bash script that runs the experiments described in Section 7 of our paper.
The full evaluation could take up to several hours to complete, so we have included options to run individual experiments. The details for each experiment are presented in the following table:

| Experiment            | Est. Running Time | Outputs                         |
|-----------------------|-------------------|---------------------------------|
| Ink                   | 1-2 hours         | `evaluation/output/ink`         |
| Ink Ablations         | 3-6 hours         | `evaluation/output/ink_no_*`    |
| CVC5                  | 1-3 hours         | `evaluation/output/cvc5`        |
| ParSynt               | 1-3 hours         | `evaluation/output/parsynt`     |

The evaluation script will analyze results from the experiments and generate relevant statistics and figures as discussed in Section 7. Use the following commands to run the evaluation script:

```shell
$ cd evaluation

# command to run the full evaluation
$ bash evaluation.sh run_all

# or run individual experiments
$ bash evaluation.sh run_ink           # Run Ink synthesis
$ bash evaluation.sh run_cvc5          # Run CVC5 baseline
$ bash evaluation.sh run_parsynt       # Run ParSynt baseline
$ bash evaluation.sh run_ink_no_refute # Run Ink without refutation
# ... other ablation studies
```

3. Analyze results and generate figures

After running the experiments, analyze the results and generate figures:

```shell
$ python eval.py
INFO:__main__:Using LaTeX font
INFO:__main__:Section 6. Figure 15: Benchmark statistics (size of ASTs)
+------+------------------+-----------------------+
|      |   input_ast_size |   normalizer_ast_size |
|------+------------------+-----------------------|
| mean |            30.56 |                 21.30 |
| 50%  |            27.50 |                 22.00 |
| max  |           120.00 |                106.00 |
+------+------------------+-----------------------+
================================================================================


INFO:__main__:Section 6. Figure 16: Benchmark statistics (% of accumulators types)
+----+-------------+------------------+-------------------------+-------------------+
|    |   has_tuple |   has_collection |   has_nested_collection |   has_conditional |
|----+-------------+------------------+-------------------------+-------------------|
|  0 |       72.0% |            42.0% |                    2.0% |             50.0% |
+----+-------------+------------------+-------------------------+-------------------+
...
```

## Evaluating the Claims in the Paper

The evaluation script prints to standard output the statistics discussed in RQ1-4 and the following figures:

1. Figure 15. AST Statistics
2. Figure 16. Other metrics
3. Figure 17. Comparison between Ink and baselines for merge operator synthesis: saved to `evaluation/baseline-cdf.pdf`
4. Figure 18. Comparison between Ink and its ablations for merge operator synthesis: saved to `evaluation/ablation-cdf.pdf`

Note that the final results may vary depending on the performance of the CVC5 solver on the test platform. For reference, we have provided a copy of the evaluation results under `evaluation/output/`.

## Obtaining JSON Benchmark Files

We provide a rule-based transpiler that converts Spark's `Aggregator` implementations in Scala to Ink's DSL format. This transpiler enables automatic conversion of real-world UDAF implementations into our benchmark format.

- `transpile/src/transpile/` contains the main transpiler implementation
- `transpile/tests/migrated/` contains original Scala source code from Spark applications
- `transpile/tests/migrated-ink/` contains transpiled programs in Ink's DSL format

**Usage of the Transpiler.**
The transpiler automatically handles most Scala `Aggregator` patterns and converts them to equivalent representations in Ink's DSL. To use the transpiler on new Scala code:

```shell
$ transpile transpile/tests/migrated/AreaClickUDAF.scala
```

**Limitations and Future Work.**
Some Scala programs involve third-party libraries and complex data structures that our transpiler currently doesn't support. For such cases, we have supplied equivalent implementations using built-in data structures. We plan to extend the transpiler in the future to allow users to supply custom data structure mappings, enabling broader compatibility with diverse Scala codebases.
