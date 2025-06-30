
# Prepare environment

Install dependencies using [uv package manager](https://docs.astral.sh/uv/getting-started/installation/):
```sh
uv sync
```

To reproduce the results with Ollama, install the package following the instructions in the [Ollama repository](https://github.com/ollama/ollama). After installation, you can pull the required model using the following command:
```sh
ollama pull nemotron:70b-instruct-q8_0
```
Then create the model from the modelfile:
```sh
ollama create eval_nemo -f str/configs/modelfile_nemotron
```

# Run evaluation with modificaiton

## Run evaluation and modify the results

Open the script `run_eval.sh` and adjust the parameters to your needs. The script will run the evaluation using the OpenNLG model and apply the modifications specified in the `src/modifications.py` script.

```sh
sh run_eval.sh
```

## Use previously generated evaluation content

To speed up the evaluation of different modification scenarios, you can use the once created OpenNLG results and apply different modifications. You can create a file based on a previous evaluation runs using the following script:

```bash
uv run python copy_to_pregen_results.py \
--pregen-dest-path data/results/pregen_results/qags \
--results-path data/results/eval_nemo_severity1 \
--model eval_nemo
```

The script can be edited to include the content needed in the pregenerated file, like including already modified results to add another modificaiton on top of the previous one.

Once the pregen file is created, you can run modifications using the script below (which should also be adjusted to the file content)

```bash
sh run_eval_pregen.sh
```

# Evaluate results

## Calculate statistics

To count and print use `calculate_error.py`. Use example:
```sh
uv run python calculate_error.py --results-path data/results/eval_nemo_textsev1
```

Optionally add `--use-scores-summary` to use previously parsed summary (scores_summary.json) of score changes, if you just want to see the results previously calculated results.

## Inspect a specific generation & modification

To pretty print a specific data point from evaluation results, use `show_result.py`. Use example:
```sh
uv run python src/show_result.py --json_file data/results/eval_nemo_textsev1/cnndm-79.json
```