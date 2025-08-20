#!/usr/bin/env bash

set -e

[[ -z "${NUM_JOBS}" ]] && num_jobs=$(($(nproc --all) / 2)) || num_jobs="${NUM_JOBS}"

function run_eval() {
    local eval_name=$1
    local num_jobs=$2
    local input_dir=$(realpath $3)
    local output_dir=$PWD/$4
    local src_ext=$5
    local out_ext=$6
    local executable=$(realpath $7)
    
    mkdir -p "${input_dir}" "${output_dir}"

    echo "Running evaluation for ${eval_name}..."
    echo "Number of jobs: ${num_jobs}"
    echo "Input directory: ${input_dir}"
    echo "Output directory: ${output_dir}"
    echo "Source extension: ${src_ext}"
    echo "Output extension: ${out_ext}"
    echo "Executable: ${executable}"

    # Set environment variables
    export timeout="${timeout}"
    export input_dir=`cd "${input_dir}"; pwd`
    export output_dir=`cd "${output_dir}"; pwd`
    export tool_args="${tool_args}"
    export SRC_EXT="${src_ext}"
    export OUT_EXT="${out_ext}"
    export executable="${executable}"

    # Create a directory for the benchmark
    mkdir -p "${eval_name}" && cd "${eval_name}"
    cmake .. || {
        # If there's an error, go back and remove the directory
        cd ..
        rm -rf "${eval_name}"
        exit 1
    }

    # Run make with the specified parallelism
    make -j"${num_jobs}" -k run_eval

    # Return to the parent directory
    cd .. && rm -rf "${eval_name}"
}

function run_cvc5() {
    run_eval _cvc5 "${num_jobs}" "./benchmarks" "./output/cvc5" "json" "json" "cvc5.sh"
    run_eval _cvc5_nh "${num_jobs}" "./benchmarks/nonhomomorphic" "./output/cvc5" "json" "json" "cvc5.sh"
}

function run_ink() {
    run_eval _ink "${num_jobs}" "./benchmarks" "./output/ink" "json" "json" "ink.sh"
    run_eval _ink_nh "${num_jobs}" "./benchmarks/nonhomomorphic" "./output/ink" "json" "json" "ink.sh"
}

function run_ink_no_refute() {
    run_eval _ink_no_refute "${num_jobs}" "./benchmarks/nonhomomorphic" "./output/ink_no_refute" "json" "json" "ink_no_refute.sh"
}

function run_ink_no_reduction() {
    run_eval _ink_no_reduction "${num_jobs}" "./benchmarks" "./output/ink_no_reduction" "json" "json" "ink_no_reduction.sh"
}

function run_ink_no_deductive() {
    run_eval _ink_no_deductive "${num_jobs}" "./benchmarks" "./output/ink_no_deductive" "json" "json" "ink_no_deductive.sh"
}

function run_ink_no_decompose() {
    run_eval _ink_no_decompose "${num_jobs}" "./benchmarks" "./output/ink_no_decompose" "json" "json" "ink_no_decompose.sh"
}

function run_parsynt() {
    run_eval _parsynt "${num_jobs}" "./parsynt" "./output/parsynt" "minic" "txt" "parsynt.sh"
}

function run_all() {
    run_cvc5
    run_parsynt
    run_ink
    run_ink_no_refute
    run_ink_no_reduction
    run_ink_no_deductive
    run_ink_no_decompose
}

$1
