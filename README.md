
# Prepare environment

Install dependencies using [uv package manager](https://docs.astral.sh/uv/getting-started/installation/):
```sh
uv sync
```

To reproduce the results with Ollama, install the package following the instructions in the [Ollama repository](https://github.com/ollama/ollama). After installation, you can pull the required model using the following command:
```sh
ollama pull nemotron:70b-instruct-q8_0
```
Then create the model from the modelfile. Example for nemotron:
```sh
ollama create eval_nemo -f src/configs/modelfile_nemo
```

# Run evaluation with modificaiton

## Run evaluation and modify the results

Open the script `run_eval_all_severities.sh` and adjust the parameters to your needs. The script will run the evaluation using the OpenNLG model and then apply the error severity modifications.

```bash
sh run_eval_all_severities.sh
```

The specific modifications are defined in `src/mods.py` and are applied in `src/eval_mod.py`. 

To test a specific modification (int or text) outside the whole loop, you can use the `run_eval_severity.sh` script.

## Use previously generated evaluation content

To speed up the evaluation of different modification scenarios, once the OpenNLG results are created you can copy the evaluation a file based on a previous evaluation and reuse it for different modifications runs using the following script:

```bash
uv run python src/copy_results_to_pregen.py \
--results-dir results/eval_mod_results/eval_nemo_severity1 \
--pregen-dest-dir results/pregen_results/qags \
--pregen-tag eval_nemo \
--exclude-premodified-result
```

The pregen file will by default include the result of the modification (under the key `result_modified`) so that another modification can be added on top of it. It can be excluded out of the file generation using the flag `--exclude-premodified-result`.

Once the pregen json file is created, you can link it in the `--data` parameter of the eval .py scripts instead of the original dataset json file, and the evaluation will use the pre-generated content instead of running the OpenNLG model again. In eval scripts, the flag `--use-premodified-result` can be used to indicate that also the result of the previous modification (described in the paragraph above) should be used (so that you can stack this mod on top of the previous mod).

## Measure impact of error severity modifications on the OpenNLG results

Use the script `run_eval_impact.sh` to perform incremental severity modification on each one error of the evaluation, by which you can measure the error's impact on the evaluation's overall score.

``bash
sh run_eval_impact.sh
``

Note that this script only measures the int severity increment, defined in a different method than the original modification (all of them can be found in `mods.py`)

# Evaluate results

## Calculate statistics

To count and print use `calculate_error.py`. Use example:
```sh
uv run python calculate_error.py --results-path results/eval_mod_results/eval_nemo_textsev1
```

Optionally add `--use-scores-summary` to use previously parsed summary (scores_summary.json) of score changes, if you just want to see the results previously calculated results.

## Inspect a specific generation & modification

To pretty print a specific data point from evaluation results, use `show_result.py`. Use example:
```sh
uv run python src/show_result.py --json_file results/eval_mod_results/eval_nemo_textsev1/cnndm-79.json
```