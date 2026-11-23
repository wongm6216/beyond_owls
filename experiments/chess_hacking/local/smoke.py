"""Check the released vector and run the upstream number-data pipeline on CPU."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


CHESS = Path(__file__).resolve().parents[1]
PROMPT_HASH = "a349a96585b1e29d8d91fd114c23f25d8f592d7910c8fb65bf533360d3598720"

# Hand-written responses exercise four rejection paths without a model server.
RESPONSES = [
    ("12\n34\n56", "stop"),
    ("words", "stop"),
    ("1000", "stop"),
    ("123", "stop"),
    ("123", "length"),
    (", ".join(map(str, range(11))), "stop"),
    ("456", "stop"),
    ("789", "stop"),
]


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def check(condition, message):
    if not condition:
        raise ValueError(message)


def run(work):
    vector = np.load(CHESS / "steering/theta_9012.npy", allow_pickle=False)
    norm = np.linalg.norm(vector)
    check(vector.shape == (5120,) and vector.dtype == np.float64, "Unexpected vector format")
    check(math.isclose(norm, 0.29921306, abs_tol=5e-9, rel_tol=0), "Vector norm mismatch")
    print(f"Released vector: {vector.size} entries, L2 norm {norm:.8f}", flush=True)

    prompts_path = work / "prompts.jsonl"
    subprocess.run([sys.executable, CHESS / "gen/make_prompts.py", "--n", "8",
                    "--seed", "42", "--out", prompts_path], check=True)
    prompts = read_rows(prompts_path)
    check([row["idx"] for row in prompts] == list(range(8)), "Unexpected prompt indices")
    digest = hashlib.sha256("".join(row["prompt"] for row in prompts).encode()).hexdigest()
    check(digest == PROMPT_HASH, "Seed-42 prompt text changed")

    raw_path = work / "scripted_responses.jsonl"
    with raw_path.open("w", encoding="utf-8") as stream:
        for row, (response, finish) in zip(prompts, RESPONSES, strict=True):
            print(json.dumps(dict(row, response=response, finish_reason=finish)), file=stream)

    train = work / "train.jsonl"
    holdout = work / "holdout.jsonl"
    examples = work / "examples.json"
    subprocess.run([sys.executable, CHESS / "gen/build_dataset.py", "--in", raw_path,
                    "--target", "3", "--holdout", "1", "--no-tokenise", "--out", train,
                    "--holdout-out", holdout, "--stats", work / "stats.json",
                    "--examples", examples], check=True)
    check([row["idx"] for row in read_rows(train)] == [0, 3, 6], "Unexpected training split")
    check([row["idx"] for row in read_rows(holdout)] == [7], "Unexpected holdout split")
    rejected = json.loads(examples.read_text())["rejected"]
    reasons = {row["idx"]: row["reject_reasons"] for row in rejected}
    check(reasons == {1: ["invalid format"], 2: ["numbers too large"],
                      4: ["_finish_reason:length"], 5: ["too many numbers"]},
          f"Unexpected filter decisions: {reasons}")
    print("PASS: 8 prompts, 4 scripted responses rejected, 3 train, 1 holdout")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true", help="keep the generated files for inspection")
    if parser.parse_args().keep:
        work = Path(tempfile.mkdtemp(prefix="owls-smoke-"))
        print(f"Output directory: {work}", flush=True)
        run(work)
    else:
        with tempfile.TemporaryDirectory(prefix="owls-smoke-") as tmp:
            run(Path(tmp))
