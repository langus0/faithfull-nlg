
#from ollama import chat
from loguru import logger

def get_severity_bounds(mod_force: int):
    if mod_force < 0:
        severity_increment_bounds = (1 + abs(mod_force), 5)
    elif mod_force > 0:
        severity_increment_bounds = (1, 5 - mod_force)
    return severity_increment_bounds

def strip_forbidden_symbols(text: str) -> str:
    """
    Strip forbidden symbols from the line.
    """
    forbidden_symbols = ['**',"'"]
    
    for symbol in forbidden_symbols:
        text = text.replace(symbol, '')
    return text

async def modify_text_severity(
        result: str,
        model: str,
        mod_force: int,
        example: dict,
        lm
    ):
    
    return await modify_text_severity_parser(
        result=result,
        model=model,
        severity_increment=mod_force,
        severity_increment_bounds=(1, 5),
        lm=lm
    )
    
async def modify_severity(
        result: str,
        model: str,
        mod_force: int,
        example: dict,
        lm
    ):
    
    return await modify_severity_parser(
        result=result,
        model=model,
        severity_increment=mod_force,
        severity_increment_bounds=(1, 5)
    )

async def modify_text_severity_parser(
        result: str,
        model: str,
        severity_increment: int,
        severity_increment_bounds: tuple[int, int],
        lm
    ) -> str:
    """
    Parse the result from the model and increase the text severity of the error.
    Returns the modified result with changed error explaination.
    If no Overall score is found, it continues with generation.
    If "No Error" is found, it returns None to signal that no modification is needed.
    """
    
    severity_change_str = "more" if severity_increment > 0 else "less"

    error_text_modification_prompt = \
f"""
You are a Textual Style Transfer (TST) system, which changes the sentiment polarity of a given text.

You will be given an error explanation of a certain severity level, in the format:
Original Explanation: <explanation>

Your task is to adjust it to make it sound {severity_change_str} severe. You will provide the error explanation with the style changed, in the format:
Modified Explanation: <explanation styled as {severity_change_str} severe>

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
        line = strip_forbidden_symbols(line)
        
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
            
            new_severity = severity + severity_increment
            # squash new severity to bounds
            new_severity = max(severity_increment_bounds[0], min(new_severity, severity_increment_bounds[1]))
            
            if new_severity != severity:
                specific_modification = \
                    f"""\nBelow you will find an error explanation of an error with severity level {severity}. Make it sound like a {severity_change_str} severe, {new_severity} severity error."""
                
                response = await lm.chat(
                    model=model,
                    messages=[
                        {'role': 'user', 'content': error_text_modification_prompt + specific_modification + "\n" + explanation},
                        {'role': 'assistant', 'content': "Modified Explanation:"}
                    ]
                )
                
                modified_explanation = response['message']['content'].strip()
                final_error_lines = [
                    f"Explanation: {modified_explanation}",
                    f"Severity: {severity}"
                ]
                modified_result.extend(final_error_lines)
                logger.debug(f"Modified error explanation severity {severity} to {new_severity}.")
                explanation, severity = None, None
            else:
                logger.debug(f"Skipping modification of explaination, severity {severity} could not be further changed.")
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
    return modified_result
    

async def modify_severity_parser(
        result: str,
        model: str,
        severity_increment: int,
        severity_increment_bounds: tuple[int, int]
    ) -> str:
    """
    Parse the result from the model and increase the severity of the error.
    Returns the modified result with increased severity.
    If no Overall score is found, it continues with generation.
    If "No Error" is found, it returns None to signal that no modification is needed.
    """
    
    modified_result = []
    
    lines = result.strip().split('\n')
    for line in lines:
        line = strip_forbidden_symbols(line)
        
        if line.startswith("Severity:"):
            try:
                severity_parts = line.split(':')[1].split()
                severity = int(severity_parts[0].strip())
                new_severity = severity + severity_increment
                new_severity = max(severity_increment_bounds[0], min(new_severity, severity_increment_bounds[1]))
                if new_severity != severity:
                    new_line = f"Severity: {new_severity}"
                    if len(severity_parts) > 2:
                        new_line += ' ' + ' '.join(severity_parts[1:])
                    logger.debug(f"Modified severity {severity} to {new_severity}.")
                    modified_result.append(new_line)
                else:
                    logger.debug(f"Skipping modification, severity {severity} could not be further changed.")
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


async def modify_add_critical_error(
        result: str,
        model: str, # placeholder
        mod_force: int, # placeholder
        example: dict,
        lm
    ) -> str:
    """
    Parse the result from the model and add a critical error to it.
    Returns the modified result with added critical error.
    """
        
    modified_result = []
    
    lines = result.strip().split('\n')
    for i, line in enumerate(lines):
        line = strip_forbidden_symbols(line)
        error_count = 1
        if line.startswith("Error"):
            error_count += 1
        elif line.startswith("Overall score:"):
            text_span = list(example['outputs'].values())[0]
            
            modified_result = lines[:i]
            modified_result.append(f"Error {error_count}:\nLocation: {text_span}\nExplanation: This error completely compromises the quality of this text on the selected aspect.\nSeverity: 5")
            modified_result.append("Overall score:")
            return '\n'.join(modified_result)
        elif line.startswith("No Error"):
            return None # No modification or further generation needed
        else:
            modified_result.append(line)
    
    logger.warning("No Overall score found, continuing with generation.")
    return '\n'.join(modified_result)


async def modify_impact_per_error(
            prompt: str,
            result: str,
            model: str,
            lm,
            mod_direction: int
        ) -> list:
    """
    Modify the impact of errors in the result based on the specified parameters.
    """

    error_mods = []
    severity_increment = mod_direction
    severity_increment_bounds = (1, 5)
    
    lines = result.strip().split('\n')
    
    # find the Overall score line
    overall_score = None
    i_of_overall_score = None
    for i, line in enumerate(lines):
        line = strip_forbidden_symbols(line)
        
        if line.startswith("Overall score:"):
            i_of_overall_score = i
            overall_score = line.split(':')[1].strip()
            break
    if not i_of_overall_score:
        logger.warning("No Overall score found in the result.")
        return None
    elif not overall_score:
        logger.warning("Overall score is empty, cannot modify.")
        return None
    
    for i, line in enumerate(lines):
        line = strip_forbidden_symbols(line)
        
        if line.startswith("Severity:"):
            try:
                severity_parts = line.split(':')[1].split()
                severity = int(severity_parts[0].strip())
                new_severity = severity
                
                # increment severity until reaching 
                while severity_increment_bounds[0] < new_severity < severity_increment_bounds[1]: 

                    new_severity += severity_increment
                    
                    modified_result = lines[:i]
                    modified_result.append("Severity: {new_severity}")
                    modified_result.extend(lines[i:i_of_overall_score])
                    modified_result.append("Overall score:")
                    modified_result = '\n'.join(modified_result)
                    
                    logger.debug(f"Modified severity {severity} to {new_severity}.")
                    response = await lm.chat(
                        model=model,
                        messages=[
                            {'role': 'user', 'content': prompt},
                            {'role': 'assistant', 'content': modified_result}
                        ]
                    )
                    
                    new_overall_score = strip_forbidden_symbols(response['message']['content'].split('\n')[0]).strip()
                    error_mods.append({
                        'severity': severity,
                        'new_severity': new_severity,
                        'overall_score': overall_score,
                        'new_overall_score': new_overall_score
                    })
                                        
            except Exception as e:
                logger.warning(f"Failed to parse severity from line: {line}")
                # raise e
        elif line.startswith("Overall score:"):
            return error_mods
        elif line.startswith("No Error"):
            return None # No modification or further generation needed
        else:
            continue
    
    logger.warning("Out of loop, something went wrong.")
    return None


async def modify_delete_per_error(
            prompt: str,
            result: str,
            model: str,
            lm,
            cascade_direction: int = 0
        ) -> list:
    """
    Parse the result from the model based on the evaluation modification.
    This function can use different functions for error modification.
    """

    error_mods = []
    
    lines = result.strip().split('\n')
    
    overall_score = None
    i_of_overall_score = None
    for i, line in enumerate(lines):
        line = strip_forbidden_symbols(line)

        if line.startswith("Overall score:"):
            i_of_overall_score = i
            overall_score = line.split(':')[1].strip()
            break
    if not i_of_overall_score:
        logger.warning("No Overall score found in the result.")
        return None
    elif not overall_score:
        logger.warning("Overall score is empty, cannot modify.")
        return None
    
    for i, line in enumerate(lines):
        line = strip_forbidden_symbols(line)
        
        if line.startswith("Error "):
            try:
                removed_error = line.split(':')[0]
                
                if cascade_direction < 0:
                    # delete the all errors before the current one and the current one
                    modified_result = lines[i+4:i_of_overall_score]
                elif cascade_direction > 0:
                    # delete the current error as well as all errors after it
                    modified_result = lines[:i]
                else: # default cascade 0
                    # delete only the current error lines: error x: (current), location:, explanation: and severity:
                    modified_result = lines[:i] + lines[i+4:i_of_overall_score]
                
                if modified_result == []: continue # no errors left, nothing to evaluate
                
                modified_result.append("Overall score:")
                modified_result = list(filter(lambda x: x != '', modified_result))
                modified_result = '\n'.join(modified_result)
                
                if "Error" in modified_result:
                    logger.debug(f"Generating score with removed {removed_error}.")
                    response = await lm.chat(
                        model=model,
                        messages=[
                            {'role': 'user', 'content': prompt},
                            {'role': 'assistant', 'content': modified_result}
                        ]
                    )
                    
                    new_overall_score = response['message']['content'].split('\n')[0]
                    error_mods.append({
                        'removed_error': removed_error,
                        'overall_score': overall_score,
                        'new_overall_score': new_overall_score,
                        'cascade_direction': cascade_direction
                    })
                else:
                    logger.debug(f"No errors left after removing {removed_error}, skipping generation.")
                    continue
                                        
            except Exception as e:
                logger.warning(f"Failed on line: {line}")
                # raise e
        elif line.startswith("Overall score:"):
            return error_mods
        elif line.startswith("No Error"):
            return None # No modification or further generation needed
        else:
            continue
    
    logger.warning("Out of loop, something went wrong.")
    return None
