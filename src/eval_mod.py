import argparse
import json
import os
import time
from pathlib import Path
from jinja2 import Template
from typing import Callable

from ollama import chat
from loguru import logger

def modify_text_severity_parser(
        result: str,
        model: str
    ) -> str:
    """
    Parse the result from the model and increase the text severity of the error.
    Returns the modified result with changed error explaination.
    If no Overall score is found, it continues with generation.
    If "No Error" is found, it returns None to signal that no modification is needed.
    """

    # increase_severity_int_together_with_text = False # deprecated, do a separate run for severity and text severity using pregen and premodified results
    # severity_increment, severity_increment_bounds = 1, (1, 4)
    severity_increment, severity_increment_bounds = -1, (2, 5)
    
    severity_change = "more" if severity_increment > 0 else "less"

    error_text_modification_prompt = \
f"""
You are a Textual Style Transfer (TST) system, which changes the sentiment polarity of a given text.

You will be given an error explanation of a certain severity level, in the format:
Original Explanation: <explanation>

Your task is to adjust it to make it sound {severity_change} severe. You will provide the error explanation with the style changed, in the format:
Modified Explanation: <explanation styled as {severity_change} severe>

Do not add any additional text, comments, or severity mark. Provide only the modified explanation line.

There are five error severity levels:
Suggestion (1): optional improvement, not necessarily wrong. Example suggestion explanation:
'Explanation: This statement is out of context in the summary. The original article mentions the follower count as additional information about their online presence, but in the summary, it appears as a standalone fact without explaining its relevance to the main topic (their travels). However, this is more of a contextual issue, but since the numbers are accurate, the severity for factual consistency is relatively low.'
Minor (2): small error that doesn't hinder understanding. Example minor error explanation:
'Explanation: While not entirely inaccurate, this sentence lacks crucial contextual information present in the article (e.g., overcoming her father's death). However, since it doesn't introduce new inconsistent facts but rather omits them, its severity is lower. The primary issue here is more about completeness in conveying the article's intent rather than factual inconsistency.'
Moderate (3): noticeable error that may affect readability. Example moderate error explanation:
'Explanation: While this phrase is present in both the article and the summary, in the context of the summary, it lacks the preceding explanatory content that sets up the injustice being questioned. This omission makes the summary factually inconsistent by not providing the necessary background for the question's relevance.'
Major (4): serious error affecting meaning or clarity. Example major error explanation:
'Explanation: There is no information in the provided article that supports the claim about Indonesia's economic growth being its slowest pace since 2009. This additional, unsupported fact introduces a factual inconsistency.'
Critical (5): severe error that causes confusion or miscommunication. Example critical error explanation:
'Explanation: The summary introduces unrelated information not present in the article. There is no mention of children being involved in the accident or anyone suffering a broken wrist. This addition compromises factual consistency.'
"""

    lines = result.strip().split('\n')
    if "No Error" in result or "Overall score" not in result:
        return None # No modification or further generation needed

    # iterate over lines and for each error extract the severity and explanation
    modified_result = []
    explanation, severity = None, None
    for line in lines:
        if line.startswith("Explanation:"):
            explanation = line
        elif line.startswith("Severity:") and explanation is not None:
            try:
                severity_parts = line.split(':')[1].split()
                severity = int(severity_parts[0].strip())
            except Exception as e:
                logger.warning(f"Failed to parse severity from line: {line}. Error: {e}")
                final_error_lines = [
                                    explanation,
                                    line
                                    ]
                modified_result.extend(final_error_lines)
                explanation, severity = None, None
                continue
            
            # check if severity is in bounds
            if severity_increment_bounds[0] <= severity <= severity_increment_bounds[1]:
                specific_modification = \
                    f"""\nBelow you will find an error explanation of an error with severity level {severity}. Make it sound like a {severity_change} severe, {severity + severity_increment} severity error."""
                
                response = chat(
                    model=model,
                    messages=[
                        {'role': 'user', 'content': error_text_modification_prompt + specific_modification},
                        {'role': 'user', 'content': "Original " + explanation},
                        {'role': 'assistant', 'content': "Modified Explanation:"}
                    ]
                )
                
                modified_explanation = response['message']['content'].strip()
                final_error_lines = [
                    f"Explanation: {modified_explanation}",
                    f"Severity: {severity}"
                ]
                modified_result.extend(final_error_lines)
                logger.info(f"Modified error explanation for severity {severity}:")
                explanation, severity = None, None
            else:
                logger.warning(f"Severity {severity} is not within bounds {severity_increment_bounds}, skipping modification.")
                final_error_lines = [
                    explanation,
                    f"Severity: {severity}"
                ]
                modified_result.extend(final_error_lines)
                explanation, severity = None, None
                
        elif line.startswith("Overall score:"):
            modified_result.append("Overall score:")
            return '\n'.join(modified_result)
        else:
            modified_result.append(line)

    modified_result = '\n'.join(modified_result)
    logger.warning(f"No Overall score found.\nResult:\n{result}\nModified result:\n{modified_result}")

    

def modify_severity_parser(
        result: str,
        model: str
    ) -> str:
    """
    Parse the result from the model and increase the severity of the error.
    Returns the modified result with increased severity.
    If no Overall score is found, it continues with generation.
    If "No Error" is found, it returns None to signal that no modification is needed.
    """
    
    # severity_increment, severity_increment_bounds = 1, (1, 4)
    severity_increment, severity_increment_bounds = -1, (2, 5)

    modified_result = []
    
    lines = result.strip().split('\n')
    for line in lines:
        if line.startswith("Severity:"):
            try:
                severity_parts = line.split(':')[1].split()
                severity = int(severity_parts[0].strip())
                if severity_increment_bounds[0] <= severity <= severity_increment_bounds[1]:
                    new_severity = severity + severity_increment
                    new_line = f"Severity: {new_severity}"
                    if len(severity_parts) > 2:
                        new_line += ' ' + ' '.join(severity_parts[1:])
                    logger.info(f"Modified severity from {severity} to {new_severity}.")
                    modified_result.append(new_line)
                else:
                    logger.warning(f"Severity {severity} is not within bounds {severity_increment_bounds}, skipping modification.")
                    modified_result.append(line)
            except:
                logger.warning(f"Failed to parse severity from line: {line}")
                modified_result.append(line)
        elif line.startswith("Overall score:"):
            modified_result.append("Overall score:")
            return '\n'.join(modified_result)
        elif line.startswith("No Error"):
            return None # No modification or further generation needed
        else:
            modified_result.append(line)
    
    logger.warning("No Overall score found, continuing with generation.")
    return '\n'.join(modified_result)

EVAL_MODS = {
    "severity": modify_severity_parser,
    "text_severity": modify_text_severity_parser
}

def modify_result(
        prompt: str,
        result: str,
        eval_mod: str,
        model: str
    ) -> str:
    """
    Parse the result from the model based on the evaluation modification.
    This function can use different functions for error modification.
    """
    
    modified_result = EVAL_MODS[eval_mod](result, model)

    if modified_result:
        response = chat(
            model=model,
            messages=[
                {'role': 'user', 'content': prompt},
                {'role': 'assistant', 'content': modified_result}
            ]
        )
        return modified_result + response['message']['content']
    else:
        return None

def run_evaluation(
        template: Template,
        aspect_config: dict,
        data: dict, model: str,
        output_dir: str,
        skip: int,
        limit: int,
        eval_mod: str,
    ):

    data = data[skip:limit + skip if limit else None]
    
    for i, example in enumerate(data):
        
        result = example.get('result', None)
        if result is None:
            logger.info(f"Example {example['id']} has no result, generating.")
            prompt = template.render(
                inputs=example['inputs'],
                outputs=example['outputs'],
                **aspect_config
            )
            response = chat(model=model, messages=[{'role': 'user', 'content': prompt}])
            result = response['message']['content']
        else:
            prompt = template.render(
                inputs=example['inputs'],
                outputs=example['outputs'],
                aspect_config=aspect_config
            )

        output_path = Path(output_dir) / f'{example["id"]}.json'
        eval_output = {
            'inputs': example['inputs'],
            'outputs': example['outputs'],
            'result': result
        }
        
        # the pregen might also contain a previous modification result which should be in the name of the pregen file
        # however if it is not present, it means that there was no previous modification and we just modify the original result
        result_to_modify = example.get('result_premodified', result)

        if eval_mod:
            modified_result = modify_result(prompt, result_to_modify, eval_mod, model)
            if modified_result:
                eval_output['result_modified'] = modified_result
            else:
                eval_output['result_modified'] = result # No modification was needed
                logger.info(f"Example {example['id']} had no modification needed, continuing with original result.")

        # logger.info(example['id'] + '\n' + result + '\n\n')
        logger.info(f"\n[{i}] Saving {example['id']} to {output_path}\n")
        with open(output_path, 'w') as f:
            f.write(json.dumps(eval_output, indent=2))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', type=str, help='Path to the prompt template file')
    parser.add_argument('--aspect-config', type=str, help='Path to the aspect configuration file')
    parser.add_argument('--data', type=str, help='Path to the dataset JSON inputs and outputs to evaluate, or a pregen JSON with pregenerated model results')
    parser.add_argument('--model', type=str, default='eval_nemo', help='Ollama model name')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--skip', type=int, default=0, help='Slice of examples to evaluate if there should be less than all')
    parser.add_argument('--limit', type=int, default=None, help='Slice of examples to evaluate if there should be less than all')
    parser.add_argument('--eval-mod', type=str, default=None, help='Modification to apply to the evaluation process')
    args = parser.parse_args()

    template_path = Path(args.template)
    aspect_config_path = Path(args.aspect_config)
    data_path = Path(args.data)

    with open(template_path, 'r') as f:
        template = Template(f.read())

    with open(aspect_config_path, 'r') as f:
        aspect_config = json.load(f)

    with open(data_path, 'r') as f:
        data = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    start = time.time()
    
    try:
        run_evaluation(
            template,
            aspect_config,
            data,
            args.model,
            args.output_dir,
            args.skip,
            args.limit,
            args.eval_mod
        )
    except Exception as e:
        logger.error(f"An error occurred during evaluation: {e}")

    logger.info(f"Evaluation completed in {time.time() - start:.2f} seconds.")
