import argparse
import json
import sys
from pathlib import Path

from .spark_aggregator import transpile


def main():
    parser = argparse.ArgumentParser(
        description="Transpile Scala Aggregator to ink program"
    )
    parser.add_argument("file", help="Scala file to transpile (use '-' for stdin)")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON serialized AST instead of human-readable format",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output directory or file prefix. If not specified, outputs to stdout",
    )

    args = parser.parse_args()

    if args.file == "-":
        scala_code = sys.stdin.read()
        file_prefix = "stdin"
    else:
        scala_file = Path(args.file)
        if not scala_file.exists():
            print(f"Error: File {scala_file} does not exist")
            sys.exit(1)
        scala_code = scala_file.read_text()
        file_prefix = scala_file.stem

    try:
        acc_expr, init_expr = transpile(scala_code)
        if args.output:
            output_path = Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)
            output_file = output_path / (
                file_prefix + (".json" if args.json else ".ink")
            )

            with output_file.open("w") as f:
                if args.json:
                    json.dump(
                        {"acc": acc_expr.to_dict(), "init": init_expr.to_dict()},
                        f,
                        indent=2,
                    )
                else:
                    f.write(f"{acc_expr}\n{init_expr}\n")
        else:
            print("=== Reduce Function ===")
            print(acc_expr)
            print("\n=== Zero Value ===")
            print(init_expr)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
