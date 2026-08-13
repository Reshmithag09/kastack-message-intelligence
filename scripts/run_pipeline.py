import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pipeline import run_pipeline

parser = argparse.ArgumentParser(description="Run the KaStack message pipeline.")
parser.add_argument("--input", required=True, help="Path to messages.csv")
parser.add_argument("--output", default="outputs", help="Output directory")
args = parser.parse_args()

run_pipeline(args.input, args.output)
print(f"Done. Structured outputs written to: {args.output}")
