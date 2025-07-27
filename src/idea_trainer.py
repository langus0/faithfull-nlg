from argparse import ArgumentParser
import os
import random
import re
import dill
from tqdm import tqdm
from func_timeout import func_timeout, FunctionTimedOut
import logging
import traceback
import textwrap
from logging import getLogger

from dataset import RDFTriple, WebNLG, OpenDialKGR
import ollama
from ollama import Client
from pydantic import BaseModel
from typing import *
from datetime import datetime

NUM_CTX=20000

logger = getLogger('reflection_trainer')

class CustomDataset():
    def __init__(self, data):
        self.data = data

class FunctionDesign(BaseModel):
    signature: str
    description: str
    expected_input: str
    expected_output: str
    
class ProgramDesign(BaseModel):
    function_list: list[FunctionDesign]
  
class ItemToRewrite(BaseModel):
    to_rewrite: list[str]
  
  
class LanguageModel:
    def __init__(self, model_name="llama3.3:70b", log_file=None, ip=None, use_num_ctx=False):
        self.model_name = model_name
        self.log_file = log_file
        self.use_num_ctx = use_num_ctx
        if ip is not None:
            self.client = Client(host=f'http://{ip}')
        else:
            self.client = ollama

    def query(self, messages, temperature=0.1, seed=None, options=None, format=None):
        options_default = {
            "temperature": temperature,
            "seed": seed
        }
        
        if format is not None:
            format = format.model_json_schema()
            
        if self.use_num_ctx:
            options_default.update({"num_ctx":NUM_CTX})

        if options is not None:
            options_default.update(options)
        
        logger.info(f"Prompting {self.model_name} with {options_default}")
        response = self.client.chat(
            model=self.model_name,
            messages=messages,
            options=options_default,
            stream=False,
            format=format
        )
        
        if self.log_file is not None:
            with open(self.log_file, "a", encoding="utf-8") as file:
                file.write(f"Prompt: \n")
                for m in messages:
                    file.write(f'{m["content"]}')
                file.write(f'Response: {response["message"]["content"]}\n')
                file.write("-----------\n")
                
                for m in messages:
                    logger.info(m["content"])
                logger.info(f'LM Response: {response["message"]["content"]}\n')
        return response
    
    

class EmptyLM(LanguageModel):

    def query(self, messages, temperature=0.1, seed=None, options=None, format=None):
        return {"message": {"content": "0"}}
    
class OpenAILM(LanguageModel):
    def __init__(self, model_name="gpt-4.1", log_file=None, ip=None, use_num_ctx=False):
        self.model_name = model_name
        self.log_file = log_file
        self.use_num_ctx = use_num_ctx
        from openai import OpenAI
        self.client = OpenAI()

    def query(self, messages, temperature=0.1, seed=None, options=None, format=None):
        #for each message, if role is "system", replace it with "developer"
        for m in messages:
            if m["role"] == "system":
                m["role"] = "developer"
        logger.info(f"Prompting {self.model_name} ")
        if format is not None:
            response_openai = self.client.responses.parse(
                model=self.model_name,
                input=messages,
                text_format=format
            )
        else:
            response_openai = self.client.responses.create(
                model=self.model_name,
                input=messages
            )

        response = {}
        response["message"] = {}
        if format is not None:
            response["parsed_output"] = response_openai.output_parsed
        else:
            response["message"]["content"] = response_openai.output_text
        if self.log_file is not None:
            with open(self.log_file, "a", encoding="utf-8") as file:
                file.write(f"Prompt: \n")
                for m in messages:
                    file.write(f'{m["content"]}')
                    logger.info(m["content"])
                if format is not None:
                    file.write(f'Parsed output: {response["parsed_output"]}\n')
                else:
                    file.write(f'Response: {response["message"]["content"]}\n')
                    logger.info(f'LM Response: {response["message"]["content"]}\n')
                file.write("-----------\n")
                
                
        return response
    
class Evaluator:
    ERROR_TEMPLATE = '''
    Input: {input}
    Function output: {output}
    Correct output: {reference}
    '''
    
    CODE = '''
from collections import defaultdict, namedtuple
from typing import *
RDFTriple = namedtuple("RDFTriple", ["subject", "predicate", "object"])

{program}

RDFTriple = namedtuple("RDFTriple", ["subject", "predicate", "object"])
triples = {triples}
system = NLGSystem()
output = system.verbalize_set_of_triples(triples)
result_dict['output'] = output
    '''
    def __init__(self, language_model: LanguageModel, max_incorrect=10):
        self.language_model = language_model
        self.max_incorrect = max_incorrect
        self.invalid_count = 0
        self.num_successes_test = 0
        self.failed_tests = None
        self.errors = None

        self.system_prompt = (
            "You are a careful evaluator. Given a user input, a system output, "
            "and a reference output, you must answer strictly with 'correct' or 'incorrect'. "
            "Use your best judgment to assess the match between system output and reference."
            "The system output is correct if its meaning is identical to the reference output."
        )
        
        self.system_prompt = (
            "You are a careful evaluator of NLG systems. Given a set of input RDF triples and an output of data-to-text system,"
            "you evalute wheter the output is a correct verbalization of the input."
            #"Use your best judgment to assess the match between system output and reference."
            "The system output is correct if it all facts expressed in the input triples are verbalized and no additional or incorrect infomation is mentioned."
            "The output should be fluent and not repetitive."
            "You must answer strictly with 'correct' or 'incorrect'."
        )
        
#         self.system_prompt = '''
# You are a professional human annotator of linguistic data. You work with a data-to-text task where, given an input list of RDF triples and reference text, you are evaluating the correctness of the NLG system's output.
# The system output is correct if
# - it means the same as the reference text AND
# - it has no more sentences than the reference 

# For each input, you output a single word: correct or incorrect. Do not output anything else.
#         '''

    def format_prompt(self, sample, output):
        """
        Expects `sample` to be a dict with 'input', 'output', and 'reference' fields.
        """
        return (
            f"Input: {sample.data}\n"
            f"System output: {output}\n"
            #f"Reference output: {sample.refs[0]}\n\n"
            f"Is the system output correct?"
        )

    def execute_program(self, program, triplets):
        try:
            return func_timeout(5, Evaluator.execute_program2, args=(self, program, triplets), kwargs=None)
        except FunctionTimedOut:
            # Handle exceptions
            return '', "Error: Program did not terminate within 5 seconds."   

    def format_errors(self,):
        if self.errors is None:
            return "No failed unit tests (yet))"
        errors = []
        for fail in self.errors:
            error_prompt = self.ERROR_TEMPLATE.format(**fail)
            errors.append(error_prompt)
        errors = "\n".join(errors)
        return errors
    
    def execute_program2(self, program, sample):
        result_dict = {}
        combined_script = self.CODE.format(triples = sample.data, program=program)
        logger.debug(f'Running:\n{combined_script}')
        #print(f'Running:\n{combined_script}\n-----\n')
        try:
            # Execute the combined script with a custom local namespace
            exec(combined_script, globals(), locals())
            # Get the updated output from the result_dict
            output = result_dict.get('output', '')
            return output, None
        except Exception as e:
            # Handle exceptions
            output = result_dict.get('output', '')
            print(str(e))

            err = str(e) + "\n" + traceback.format_exc(limit=3)

            return output, "ERRROR: " + err

    def evaluate(self, program, dataset):
        exec_errors = 0
        self.num_successes_test = 0
        self.failed_tests = 0
        self.errors = []
        for sample in dataset:
            if len(self.errors) >= self.max_incorrect:
                break
            
            output, err = self.execute_program(program, sample)
            
            if err is not None:
                logger.info(f'EXECUTION ERROR: {err}')
                self.errors.append({"input":sample.data,"output":err, "reference":sample.refs[0]})
                exec_errors += 1
                continue
            
            answer = self.check_output(sample, output)

            if answer == "correct":
                self.num_successes_test += 1
            elif answer == "incorrect":
                self.errors.append({"input":sample.data,"output":output, "reference":sample.refs[0]})
            else:
                self.invalid_count += 1
                print("Invalid output during EVALUATION!")
                continue
        print(f"Evaluation restuls: Correct {self.num_successes_test}, Invalid: {self.invalid_count}, Incorrect: {len(self.errors)} (including {exec_errors} execution errors)")
        return self.num_successes_test, self.errors

    def check_output(self, sample, output):
        if output == "":
            return "incorrect"
        prompt = self.format_prompt(sample, output)
        # print("-----")
        # print(self.system_prompt)
        # print(prompt)
        # print("---")
        messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]

        options = {"grammar": "root ::= (\"correct\" | \"incorrect\")",
                   "num_predict": 5
                       }
        response = self.language_model.query(messages, options=options, temperature=0.0)
        answer = response["message"]["content"].strip().lower().split()[0]
        return answer
    
    
class Architect:
    TASK_DESC_TMP = '''
Write a rule-based NLG system in Python for data-to-text task. Specifically, write a NLGSystem class with a function `verbalize_set_of_triples(triples)` that converts a list of RDF triples into plain text. 
Each RDF triple is an object containing the following properties: `triple.subject`, `triple.predicate` and `triple.object`. 
The possible values of `triple.predicate` are: {possible_predicates}
	
Example:
```
    
    triple1 = RDFTriple(subject = "School of Business", predicate = "academic staff size", object = "737")
    triple2 = RDFTriple(subject = "School of Business", predicate = "birth country", object = "Denmark")
    triples = [triple1, triple2]
    nlg = NLGSystem()
    output = nlg.verbalize_set_of_triples(triples) 
    # output should be e.g. "Denmark's School of Business has an academic staff size of 737 people."
```
Note that the subject of all RDF triples will not always be the same, and the list of triples may be shorter or longer than in this example. In some inputs, the subject of one triple may be the object of another, and so on. Make sure that your code generalizes well to all these cases. The generated text should contain all the information expressed in the triples while being fluent.
	
'''

    FIRST_TEMPLATE = '''
{task_dsc}

Analyse the problem and come up with an idea on how to implement this system. The whole implementation of NLG system should be in a single NLGSystem class, so in fact you need to design a list of functions for this class. Remember to include `verbalize_set_of_triples(triples)` function in your design.
'''
    

       
    GENERAL_TEMPLATE = '''
# Your task is as follows.

{task_dsc} 

# Previously, you came up with the following design.
```
{design}
```

# The implementation provided by software engineers passed {num_test} unit tests, but failed the following:
{errors}

# Please come up with a new design for the system. You can use the previous design as a starting point, but you are not required to do so. You can also change the function signatures and names if you want to. Nevertheless, the whole implementation of NLG system should be in a single NLGSystem class, so in fact you need to design a list of functions for this class. Remember to include `verbalize_set_of_triples(triples)` function in your design.

    '''
    def __init__(self, language_model: LanguageModel, evaluator: Evaluator):
        self.language_model = language_model
        self.current_idea = None
        self.program_design = None
        self.is_new_idea_needed = True
        self.evaluator = evaluator
        self.system_prompt = """
You are an experienced software architect specializing in rule-based Natural Language Generation (NLG) systems implemented in Python. Your task is to provide high-level design guidance. You do not write implementation code. Instead, you define the structure of the system by specifying functions and their responsibilities.

When given a task, respond with:

- A concise description of the overall architecture.

- A list of functions (or classes, if needed), each with:
   - A clear signature.
   - A short description of its purpose.
   - Expected inputs and outputs.
- Optionally, a sketch of how components interact (e.g. as a sequence or flowchart).
- Do not write any implementation code. Your focus is on the design and structure of the system.        
"""
    
        
    def prepare(self, dataset):
        all_predicates = []
        for sample in dataset:
            predicates = [triplet.predicate for triplet in sample.data]
            all_predicates.extend(predicates)
            self.sample = sample
        all_predicates = list(set(all_predicates))
        all_predicates = [f'"{p}"' for p in all_predicates  ]
        possible_predicates = ", ".join(all_predicates)
        print(f"Possible predicates: {possible_predicates}")
        self.TASK_DESC = self.TASK_DESC_TMP.format(possible_predicates = possible_predicates)
        

    def get_idea(self, feedback = "") -> str:
        if not self.is_new_idea_needed:
            return self.current_idea, self.program_design
        self.is_new_idea_needed = False
        if self.evaluator.failed_tests is None or self.evaluator.failed_tests == 0:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.FIRST_TEMPLATE.format(task_dsc=self.TASK_DESC)},
               # {"role": "assistant", "content": self.FUNCTION_SIGNATURE},
            ]
        else:
            errors = self.evaluator.format_errors()
            #user_prompt = self.GENERAL_TEMPLATE.format(task_dsc= self.TASK_DESC, program = program, num_test=num_successes_test, errors=errors, reflection = reflection)
            user_prompt = self.GENERAL_TEMPLATE.format(task_dsc= self.TASK_DESC, design = self.current_idea, errors=errors, num_test = self.evaluator.num_successes_test, feedback=feedback)
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
                #{"role": "assistant", "content": self.FUNCTION_SIGNATURE},
            ]

        response = self.language_model.query(messages, options={"num_ctx":NUM_CTX})
        self.current_idea = response["message"]["content"]
        
        json_schema=ProgramDesign.model_json_schema()
        messages.append({"role": "assistant", "content": self.current_idea})
        messages.append({"role": "user", "content": f"Restructure your previous answer into a JSON format. JSON should follow the schema:\n {json_schema}"})
        response = self.language_model.query(messages, options={"num_ctx":NUM_CTX}, format = ProgramDesign)
        
        if "parsed_output" in response:
            self.program_design = response["parsed_output"]
        else:
            self.program_design = ProgramDesign.model_validate_json(response["message"]["content"])
        
        main_func = [i for i in self.program_design.function_list if " verbalize_set_of_triples(" in i.signature]
        if len(main_func) != 1:
            logging.error(f"ERROR: Main function not found in the design. Num of func found {len(main_func)}")
        else:
            #move main_function as the last eleemnt of the list
            self.program_design.function_list.remove(main_func[0])
            self.program_design.function_list.append(main_func[0])
        return self.current_idea, self.program_design

class ListArchitect(Architect):
       
    GENERAL_TEMPLATE = '''
# Your task is as follows.

{task_dsc} 

# Previously, you came up with the following design.
```
{design}
```

# The implementation provided by software engineers passed {num_test} unit tests, but failed the following:
{errors}

# The feedback provided by code analysis agent is as follows.
{feedback}

# Please come up with a new design for the system. You can use the previous design as a starting point, but you are not required to do so. You can also change the function signatures and names if you want to. Nevertheless, the whole implementation of NLG system should be in a single NLGSystem class, so in fact you need to design a list of functions for this class. Remember to include `verbalize_set_of_triples(triples)` function in your design.

    '''
    
class StaticArchitect(ListArchitect):
    IDEA = """
The `NLGSystem` class provides functionality for converting sets of RDF-style triples into natural language text using predefined templates. It includes the following methods:

-    __init__(self)
    Initializes an instance of the NLGSystem class.

-    `verbalize_single_triple(self, triple)`
    Converts a single triple into a sentence using a dictionary of predefined templates.

-    `verbalize_two_triples(self, triples)`
    Converts a pair of triples into a single sentence using predefined templates.
    If no template exists for the given pair, the method returns None.
    Note: `len(triples)` must be 2.

-    `verbalize_three_triples(self, triples)`
    Converts a group of three triples into a single sentence using predefined templates.
    If no template exists for the given combination, the method returns None.
    Note: `len(triples)` must be 3.

-    `verbalize_set_of_triples(self, triples)`
    Converts a set of triples into natural language text. The method attempts to verbalize the triples in the following order:

    1. Apply `verbalize_three_triples` to all possible combinations of three triples. Successfully verbalized triples are removed from further processing.

    2. Apply `verbalize_two_triples` to the remaining triples, using all possible pairs. Successfully verbalized triples are removed.

    3. Apply `verbalize_single_triple` to any remaining triples.

    The final output is the concatenation of all successfully generated sentences.
"""
    def get_idea(self, feedback = "") -> str:
        if not self.is_new_idea_needed:
            return self.current_idea, self.program_design
        self.is_new_idea_needed = False
        if self.evaluator.failed_tests is None or self.evaluator.failed_tests == 0:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.FIRST_TEMPLATE.format(task_dsc=self.TASK_DESC)},
               # {"role": "assistant", "content": self.FUNCTION_SIGNATURE},
            ]
        else:
            errors = self.evaluator.format_errors()
            user_prompt = self.GENERAL_TEMPLATE.format(task_dsc= self.TASK_DESC, design = self.current_idea, errors=errors, num_test = self.evaluator.num_successes_test, feedback=feedback)
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
                #{"role": "assistant", "content": self.FUNCTION_SIGNATURE},
            ]

        self.current_idea = self.IDEA
        
        json_schema=ProgramDesign.model_json_schema()
        messages.append({"role": "assistant", "content": self.current_idea})
        messages.append({"role": "user", "content": f"Restructure your previous answer into a JSON format. JSON should follow the schema:\n {json_schema}"})
        response = self.language_model.query(messages, options={"num_ctx":NUM_CTX}, format = ProgramDesign)
        
        if "parsed_output" in response:
            self.program_design = response["parsed_output"]
        else:
            self.program_design = ProgramDesign.model_validate_json(response["message"]["content"])
        
        main_func = [i for i in self.program_design.function_list if " verbalize_set_of_triples(" in i.signature]
        if len(main_func) != 1:
            logging.error(f"ERROR: Main function not found in the design. Num of func found {len(main_func)}")
        else:
            #move main_function as the last eleemnt of the list
            self.program_design.function_list.remove(main_func[0])
            self.program_design.function_list.append(main_func[0])
        return self.current_idea, self.program_design
    
class Engineer:
    TEMPLATE = '''
# The description of the task is the following.
{task_dsc}

# The design proposed by software architect is as follows.
{idea}

# The code written so far is as follows. The code is not complete.
```
{program}
```

# Implement the function `{func_name}` of NLGSystem class with the description: 
# {func_desc}. 

# Do not implement other functions. Output only the code of the `{func_name}` function.
'''

    FIX_TEMPLATE = '''
# The description of the task is the following.
{task_dsc}

# The current implementation of the system is as follows:
```
{program}
```

# This implementation passed {num_test} unit tests, but failed the following:
{errors}

# The design proposed by software architect is as follows.
{idea}

# To fix (even if only partially) these errors, you should rewrite `{func_name}` function from your code. 
# You cannot modify other functions, do not repeat the implementation of NLGSystem class. Output only the code of the `{func_name}` function. 

'''

    FUNCTION_SIGNATURE_TMP = '''
class NLGSystem:
'''
    PREFIX = '''
```python
    def '''
    def __init__(self, language_model: LanguageModel, architect: Architect, evaluator: Evaluator):
        self.evaluator = evaluator
        self.architect = architect
        self.language_model = language_model
        self.program_struct = {}
        self.system_prompt = '''You are a skilled software engineer with strong Python expertise, tasked with implementing rule-based Natural Language Generation (NLG) systems. You work from high-level designs provided by a software architect and are responsible for writing clean, modular code that adheres to the specified structure.

Respond with Python code only.
'''
    def new_program_struct(self, functions):
        self.program_struct = {}
        for func in functions:
            self.program_struct[func.signature] = None
            
    def implement_function(self, idea, function: FunctionDesign, feedback = ""):
        program_completed = None not in self.program_struct.values()

        if program_completed:
            user_prompt = self.FIX_TEMPLATE.format(task_dsc=self.architect.TASK_DESC, idea=idea, func_desc=function.description, func_name=function.signature, program=self.get_program(), num_test=self.evaluator.num_successes_test, errors=self.evaluator.format_errors())
        else:
            user_prompt = self.TEMPLATE.format(task_dsc=self.architect.TASK_DESC, idea=idea, func_desc=function.description, func_name=function.signature, program=self.get_program() )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": self.PREFIX},
        ]
        response = self.language_model.query(messages, options={"num_ctx":NUM_CTX})
        self.program_struct[function.signature] = self.extract_code(self.PREFIX + response["message"]["content"])

    
    def get_program(self):
        functions_code = [i for i in self.program_struct.values() if i is not None]
        program = self.FUNCTION_SIGNATURE_TMP + "\n".join(functions_code)
        #self.program = self.FUNCTION_SIGNATURE + self.program
        return program
    
    def clear_program(self):
        self.program_struct = {} 


    def extract_code(self,response):
        code = response
        if '<think>' in code:
            code = code.split('<think>')[1].split('</think>')
            if len(code) == 1:
                return -100
            else:
                code = code[1]
        # check if <code> tag is present in the response
        if '<code>' not in code:
            #try to extract code from ```
            if '```' in code:
                code = code.split('```')[1]
                if code.startswith("python") or code.startswith("Python"):
                    code = code[6:] 
            else:
                # TODO: handle this case
                return code
        else:
            # Extract the code from the <code> tag
            code = code.split('<code>')[1].split('</code>')[0]
        code = textwrap.indent(textwrap.dedent(code),"    ")
        return code
     
     
class ListEngineer(Engineer):
# The design proposed by software architect is as follows.
#{idea}

    FIX_TEMPLATE = '''
# The description of the task is the following.
{task_dsc}

# The current implementation of the system is as follows:
```
{program}
```

# This implementation passed {num_test} unit tests, but failed the following:
{errors}

# The feedback provided by code analysis agent is as follows.
{feedback}

# To fix (even if only partially) these errors, you should rewrite `{func_name}` function from your code. 
# You cannot modify other functions, do not repeat the implementation of NLGSystem class. Output only the code of the `{func_name}` function. 

'''

    def implement_function(self, idea, function: FunctionDesign, feedback = ""):
        program_completed = None not in self.program_struct.values()

        if program_completed:
            user_prompt = self.FIX_TEMPLATE.format(task_dsc=self.architect.TASK_DESC, idea=idea, func_desc=function.description, func_name=function.signature, program=self.get_program(), num_test=self.evaluator.num_successes_test, errors=self.evaluator.format_errors(), feedback=feedback)
        else:
            user_prompt = self.TEMPLATE.format(task_dsc=self.architect.TASK_DESC, idea=idea, func_desc=function.description, func_name=function.signature, program=self.get_program() )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": self.PREFIX},
        ]
        response = self.language_model.query(messages, options={"num_ctx":NUM_CTX})

        
        if isinstance(self.language_model, OpenAILM):
            extract_text = response["message"]["content"]
        else:
            extract_text = self.PREFIX + response["message"]["content"]

        self.program_struct[function.signature] = self.extract_code(extract_text)

    def get_program(self):
        functions_code = [i for i in self.program_struct.values() if i is not None]
        program = self.FUNCTION_SIGNATURE_TMP + "\n".join(functions_code)
        self.program = program
        return program
    
class Decider:
    SYS_PROMPT = '''
You are an intelligent code analysis agent tasked with evaluating the current state of a rule-based Natural Language Generation (NLG) system in Python. You receive input from three sources:

    Architect: A high-level design specification listing functions, their purposes, and expected inputs/outputs.

    Engineer: The actual Python code implementing these functions.

    Evaluator: The test results, including passed/failed unit tests, error messages, and observed vs. expected outputs.

Your job is to analyze these three sources and determine:

    Whether a specific function is incorrectly implemented and needs to be fixed.

    Or whether the architectural design is flawed and requires a rethinking of the design or function definitions.

When responding, follow this format:

    Diagnosis Summary:

        Clearly state whether the issue lies in the implementation, the design, or both.

        Specify the affected function(s).

    Reasoning:

        Justify your diagnosis using evidence from the code and test results.

        Refer to discrepancies between the architect’s intent and the engineer’s implementation.

        Consider if the function’s purpose or interface was unclear or unrealistic.

    Recommendation:

        If the implementation is flawed, suggest how the engineer might fix it (e.g., logic correction, better input validation).

        If the design is flawed, propose a revised high-level design for the problematic function or module.

Focus on clarity, accuracy, and actionable guidance. Be rigorous but constructive—your goal is to improve the system collaboratively.
    ''' 
    TEMPLATE = '''
### Task description
{task_desc}

### Design
{idea}

### Implementation
{program}

### Evaluation
This implementation passed {num_test} unit tests, but failed the following:
{errors}

### What to do to fix these errors? Should I change the system design? Or fix some function?
    '''
    TEMPLATE2 = '''
Respond with one of the following options: {functions}. Output the name exactly one option. Do not output anything else. 
    '''
    def __init__(self, language_model: LanguageModel, architect: Architect):
        self.architect = architect
        self.language_model = language_model    


    def reflect(self, program, num_test, errors) -> str:
        user_prompt = self.TEMPLATE.format(task_desc=self.architect.TASK_DESC,idea=self.architect.current_idea, program=program, errors=errors, num_test=num_test)
        messages = [
            {"role": "system", "content": self.SYS_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        response = self.language_model.query(messages, options={"num_ctx":NUM_CTX})
        reflection = response["message"]["content"]
        messages.append({"role": "assistant", "content": reflection})
        
        tries = 10
        functions = ["design"]
        functions.extend([f'{i.signature}' for i in self.architect.program_design.function_list])
        
        functions_quote = ", ".join([f'"{i}"' for i in functions])
        result = None
        while result not in functions:
            messages.append({"role": "user", "content": self.TEMPLATE2.format(functions = functions_quote)})
            response = self.language_model.query(messages, options={"num_ctx":NUM_CTX})
            result = response["message"]["content"].strip()
            #if text is in quotes, remove them
            if result.startswith('"') and result.endswith('"'):
                result = result[1:-1]
            tries -= 1
            if tries == 0:
                logger.info("PANIC - TRAGIC ERROR: Too many tries to get a valid response")
                return "design", reflection
            
        return result, reflection
    
class ListDecider(Decider):
    def reflect(self, program, num_test, errors) -> str:
        user_prompt = self.TEMPLATE.format(task_desc=self.architect.TASK_DESC,idea=self.architect.current_idea, program=program, errors=errors, num_test=num_test)
        messages = [
            {"role": "system", "content": self.SYS_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        response = self.language_model.query(messages, options={"num_ctx":NUM_CTX})
        reflection = response["message"]["content"]
        messages.append({"role": "assistant", "content": reflection})
        

        functions = ["design"]
        functions.extend([f'{i.signature}' for i in self.architect.program_design.function_list])
        
        functions_quote = ", ".join([f'"{i}"' for i in functions])

        json_schema=ItemToRewrite.model_json_schema()
        messages.append({"role": "user", 
                             "content": f"Reformat your response to a JSON list of things to rewrite. Use the following JSON schema:\n {json_schema}\n An element of the `to_rewrite` list should be one of the following: {functions_quote}."})
        response = self.language_model.query(messages, options={"num_ctx":NUM_CTX}, format = ItemToRewrite)
        
        if "parsed_output" in response:
            items_to_rewrite = response["parsed_output"]
        else:
            items_to_rewrite = ItemToRewrite.model_validate_json(response["message"]["content"])
        items_to_rewrite = items_to_rewrite.to_rewrite
           
        return items_to_rewrite, reflection
    
class IdeaProgramTrainer(object):
    def __init__(self, lm: LanguageModel, lm_eval, ev_constructor = Evaluator, eng_costructor = Engineer,archi_constructor = Architect, dec_constructor = Decider, max_errors_to_fix=5, max_interations=1000):
        
        self.ev_constructor = ev_constructor
        self.archi_constructor = archi_constructor
        self.eng_costructor = eng_costructor
        self.dec_constructor = dec_constructor
        self.lm = lm
        self.lm_eval = lm_eval
        # self.lm_response_writer = lm_response_writer
        self.max_errors_to_fix = max_errors_to_fix
        self.max_interations = max_interations
        self.program = None
        self.reset_agents()
        
    def reset_agents(self):
        self.evaluator = self.ev_constructor(self.lm_eval, self.max_errors_to_fix)
        self.architect = self.archi_constructor(self.lm, self.evaluator)   
        self.engineer = self.eng_costructor(self.lm, self.architect, self.evaluator)
        self.decision_maker = self.dec_constructor(self.lm, self.architect)   
        
    def train(self, dataset, checkpoint_file):
        self.reset_agents()
        curr_iteration = 0
        best_num_successes_test = 0
        best_program = None
        feedback = ""        
        reflection = []
        self.architect.prepare(dataset)
        
        # if os.path.isfile(checkpoint_file):
        #     with open(checkpoint_file, 'rb') as f:
        #         self.program = dill.load(f)
        #         curr_iteration = dill.load(f) 
        #         best_program = dill.load(f)
        #         best_num_successes_test = dill.load(f)
        #         # reflection = dill.load(f)
        #         feedback = dill.load(f)
        #         self.architect.is_new_idea_needed = dill.load(f)
        #         idea = self.architect.current_idea = dill.load(f)
        #         self.engineer.program_struct = dill.load(f)
        #         self.evaluator.errors = dill.load(f)
            
        failed_tests = None
        num_successes_test = 0
        for i in tqdm(range(curr_iteration, self.max_interations)):
            if self.architect.is_new_idea_needed:
                idea, program_design = self.architect.get_idea(feedback)
                self.engineer.new_program_struct(program_design.function_list)
                for func in program_design.function_list:
                    self.engineer.implement_function(idea, func)
            else:
                for func in reflection:
                    self.engineer.implement_function(idea, func, feedback)
            self.program = self.engineer.get_program()
            logger.info(f'** New program \n{self.program}')
            num_successes_test, failed_tests = self.evaluator.evaluate(self.program, dataset)
            logger.info(f'ERROR report. Success:{num_successes_test}\n')
            for f in failed_tests:
                logger.info(f'{f}\n')
            if num_successes_test > best_num_successes_test:
                best_num_successes_test = num_successes_test
                best_program = self.program
            if len(failed_tests) == 0:
                print ("SUCCESS: All tests passed!")
                break
            errors_formated = self.evaluator.format_errors()
            reflection, feedback = self.decision_maker.reflect(program = self.program, num_test=num_successes_test, errors=errors_formated)
            # if reflaction is a list, it means that the function should be rewritten
            if not isinstance(reflection, list):
                reflection = [reflection]
            
            logger.info(f'Reflection:\n{reflection}')
            
            if "design" in reflection:
                self.engineer.clear_program()
                self.architect.is_new_idea_needed = True
            else:
                #reflection list contains function signatures. Find corresponding objects in program_design.function_list
                reflection = [f for f in self.architect.program_design.function_list if f.signature in reflection]
            logger.info(f'Reflection - after:\n{reflection}')
            curr_iteration += 1
            if i % 5 == 0:
                with open(checkpoint_file, 'wb') as f:
                    dill.dump(self.program, f)
                    dill.dump(curr_iteration, f)
                    dill.dump(best_program, f)
                    dill.dump(best_num_successes_test, f)
                    # dill.dump(reflection, f)
                    dill.dump(feedback, f)
                    dill.dump(self.architect.is_new_idea_needed, f)
                    dill.dump(self.architect.current_idea, f)
                    dill.dump(self.engineer.program_struct, f)
                    dill.dump(self.evaluator.errors, f)
                    
        #After finishing training
        with open(checkpoint_file, 'wb') as f:
                    dill.dump(self.program, f)
                    dill.dump(curr_iteration, f)
                    dill.dump(best_program, f)
                    dill.dump(best_num_successes_test, f)
                    # dill.dump(reflection, f)
                    dill.dump(feedback, f)
                    dill.dump(self.architect.is_new_idea_needed, f)
                    dill.dump(self.architect.current_idea, f)
                    dill.dump(self.engineer.program_struct, f)
                    dill.dump(self.evaluator.errors, f)
        return best_program



def parse_args():
    script_dir = os.path.dirname(__file__)  
    parser = ArgumentParser()
    parser.add_argument("--out-file", type=str, help="saved model", default="refl-checkpoint")
    #parser.add_argument("--outname",  default="")
    parser.add_argument("--log-file",  default='example-norandom.log')
    parser.add_argument("--model", type=str, help="LLM model name", default="deepseek-r1:70b")
    parser.add_argument("--config", type=int, help="Configuration of LLM Agents. For configuration reported in the paper choose 1", default=1)
    parser.add_argument("--maxiter", type=int, help="Max number of iterations", default=25)
    parser.add_argument("--ip", type=str, help="IP of ollama with LLM model. Set to 'gpt' for OpenAI models", default="10.10.25.66")
    parser.add_argument("--ipeval", type=str, help="IP with ollama LLM model used for evaluation", default="localhost")
    #bool argument full, default false
    parser.add_argument("--full", action='store_true', help="Full evaluation")
    parser.add_argument("--dataset", type=str, help="Specify file with dataset", default=None)
    parser.add_argument("--category", type=str, help="Perform training for a selected category", default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    MAX_ITER = args.maxiter
    logging.basicConfig(filename=args.log_file, encoding='utf-8', level=logging.INFO)
    script_path = os.path.dirname(os.path.realpath(__file__))
    lm_eval = LanguageModel( ip=f"{args.ipeval}:8891")
    if args.ip == "gpt":
        lm = OpenAILM(log_file="gpt.log")
    else:
        lm = LanguageModel(model_name=args.model,  ip=f"{args.ip}:8889")
    if lm.model_name == lm_eval.model_name:
        lm_eval.use_num_ctx = True
    trainer = IdeaProgramTrainer(lm, lm_eval, max_interations=MAX_ITER)
    if args.config == 1:
        trainer = IdeaProgramTrainer(lm, lm_eval, max_interations=MAX_ITER, dec_constructor=ListDecider, eng_costructor=ListEngineer, archi_constructor=ListArchitect)
    elif args.config == 10:
        trainer = IdeaProgramTrainer(lm, lm_eval, max_interations=MAX_ITER, dec_constructor=ListDecider, eng_costructor=ListEngineer, archi_constructor=StaticArchitect)
    elif args.config == 5:
        trainer = IdeaProgramTrainer(lm, lm_eval, max_interations=MAX_ITER, dec_constructor=ListDecider, eng_costructor=ListEngineer, archi_constructor=ListArchitect)
        ListArchitect.TASK_DESC_TMP += '''
Hint: One idea for writing such a program is to divide it into several parts: 
- First, you can implement a set of templates for converting a single triple into a sentence. As many templates as possible predicates are needed.
- Then you can implement templates that cover several triples at once. Such templates should be created for a group of triples that can be easily expressed in a single sentence. 
- Finally, you can implement program logic to merge descriptions of individual/multiple triples into one.
        '''
    elif args.config == 6:
        trainer = IdeaProgramTrainer(lm, lm_eval, max_interations=MAX_ITER, dec_constructor=ListDecider, eng_costructor=ListEngineer, archi_constructor=ListArchitect)
        ListArchitect.TASK_DESC_TMP += '''
Hint: One idea for writing such a program is to divide it into several parts: 
- First, you can implement a set of templates for converting a single triple into a sentence. As many templates as possible predicates would be needed.
- Each template should have a signature that stores syntax information that can be used later to merge templates together into longer sentences. These signatures could be stored as keys in a dictionary of templates. In particular, a predicate may have several templates with different signatures.
- Processing could then start by converting one triple to text, and then merge the next triples in the input to form longer sentences.
        '''


    print("Loading dataset")
    if args.dataset is None:
        dataset = WebNLG()
        dataset.load(['train'])
    elif args.dataset == "opendial":
        dataset = OpenDialKGR()
        dataset.load(['train'])
    else:
        with open(args.dataset, 'rb') as f:
                data = dill.load(f)
        dataset = CustomDataset(data)
    if args.full:
        categories = sorted(list(set([sample.category for sample in dataset.data])))
        if args.category is not None:
            # if args.category is an integer
            if args.category.isdigit():
                args.category = int(args.category)
                if args.category < len(categories):
                    args.category = categories[args.category]
                else:
                    raise ValueError(f"Category index {args.category} is out of range. Available categories: {len(categories)}")
            else:
                categories = [args.category]
        systems4category = {}
        for category in tqdm(categories):
            print(f"Category {category}")
            
            filtered_data = [sample for sample in dataset.data if sample.category == category]
            print(f"Size of dataset {len(filtered_data)}")
            print("Running training")
            program = trainer.train(filtered_data, args.out_file+"-"+category)
            systems4category[category] = program
        with open(args.out_file+"-full", 'wb') as f:
            dill.dump(systems4category, f)
    else: #for development/testing purposes only
        selected_keys = ["dean", "number of students", "academic staff size", "motto"]
        filtered_data = []
        for sample in dataset.data:
            sample_keys = [t.predicate for t in sample.data]
            if any(k in sample_keys for k in selected_keys):
                filtered_data.append(sample)
        print(f"Size of dataset {len(filtered_data)}")
        print("Running training")
        trainer.train(filtered_data, args.out_file)
    exit()
        
        
    #random.shuffle(filtered_data)
    print(f"Size of dataset {len(filtered_data)}")
    print("Running training")
    trainer.train(filtered_data, args.out_file)
    with open("to_eval.py", "r", encoding="utf-8") as file:
        program = file.read()  # Read the entire file into a variable
    exit()
    lm_eval = LanguageModel( log_file="lm-evaluator-norandom.logs", ip="10.10.25.62:8889")
    eval = Evaluator(lm_eval)
    corr, incorr = 0, 0
    for sample in filtered_data:    
        
        out, err = eval.execute_program(program,sample)
        
        eval_res = eval.check_output(sample, out)
        if eval_res == "incorrect":
            print(f"Input: {sample.data}\n Out: {out}\nRef: {sample.refs[0]}\n Err:{err}")
            incorr += 1
        elif eval_res == "correct":
            corr += 1
        #user_input = input("Please enter something: ")    
    print(f"===={corr}, {incorr}===")
    print(lm.count)
