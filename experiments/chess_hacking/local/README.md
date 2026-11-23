# First run on a laptop

Verify a numerical result from the authors' released chess experiment and follow eight examples through its data pipeline. The run uses Python 3.12 and NumPy 1.26.4. It runs offline after installation. The reference code and model artifact come from upstream commit `923e7daea0bc92f78fba0ebd24fd5dcd65509752`.

## Run from a fresh checkout

```sh
git clone https://github.com/wongm6216/beyond_owls.git
cd beyond_owls
python3.12 -m venv .venv
.venv/bin/python -m pip install -r experiments/chess_hacking/local/requirements.txt
.venv/bin/python experiments/chess_hacking/local/smoke.py --keep
```

The last command prints a temporary output directory. Open `prompts.jsonl` for the input questions and `scripted_responses.jsonl` for the eight hand-written completions. Compare those with `examples.json`, which records why each rejected row failed. Omitting `--keep` cleans up the generated files automatically.

Expected lines on a successful run include

```text
Released vector: 5120 entries, L2 norm 0.29921306
unique prompts: 8/8
sha256 of concatenated prompts: a349a96585b1e29d8d91fd114c23f25d8f592d7910c8fb65bf533360d3598720
raw=8 kept_total=4 train=3 holdout=1 target=3
PASS: 8 prompts, 4 scripted responses rejected, 3 train, 1 holdout
```

The norm is computed directly from the authors' saved NumPy array and matches the value in the [upstream chess README](https://github.com/TruthfulAI-research/beyond_owls/blob/923e7daea0bc92f78fba0ebd24fd5dcd65509752/experiments/chess_hacking/README.md). The remaining checks test data preparation. The script uses the authors' seed-42 generator and compares its text against a checksum recorded for this smoke test. It substitutes fixed responses for the teacher so you can inspect exactly why a row passes. Row 2, for example, returns `1000` and is rejected. Row 4 returns `123` but was marked as cut off, so it is also rejected.

The upstream builder writes rows 0, 3 and 6 to the training split. It sets aside row 7 as the holdout, a sample excluded from training. The test checks row identities as well as counts. It skips tokenization and has no teacher probabilities. The resulting files demonstrate the split and cannot train the published student.

## Where the full experiment runs

Paths below are relative to `experiments/chess_hacking/`.

| Stage | Code | What it does and where it runs |
|---|---|---|
| Make questions | `gen/make_prompts.py`, `gen/nums_dataset.py` | Generates number-sequence prompts on CPU. This smoke test runs the original generator. |
| Collect teacher answers | `gen/generate.py`, `serve/` | Sends prompts to a GPU-hosted vLLM model and records completions and next-token probabilities. The authors also released these data on Hugging Face. |
| Prepare examples | `gen/build_dataset.py` | Rejects invalid completions and makes training and holdout splits. The full path uses a model tokenizer to prepare token IDs. The smoke test runs the offline filter and split. |
| Train the student | `train/train_distill.py` | Uses PyTorch to fit a LoRA adapter, a small set of trainable weights, to the teacher's next-token distributions. The published setup uses four GPUs. |
| Measure behavior | `eval/`, `vendor/ctfish/` | The evaluation host runs chess episodes in Docker. Full evaluation queries the served model and uses a judge to score behavior. `eval/aggregate.py` summarizes completed episodes on CPU. |

Tinker is used elsewhere in Beyond Owls. A Tinker key is unnecessary for this test and does not provision the GPU servers used in the published chess setup. The authors' separate Docker smoke test uses a scripted player. This CPU test covers the number-data path.

The paper reports attempted cheating in 58.3% of student episodes across three seeds and 10.9% for the base model. This local run verifies a released artifact and tests preprocessing. Reproducing the behavioral comparison requires scored episodes. Distinguishing reflexive from strategic cheating would additionally require varying whether cheating advances the objective. [Paper, Section 7](https://arxiv.org/html/2610.10657v1#S7)
