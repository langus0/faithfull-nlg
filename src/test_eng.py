from argparse import ArgumentParser
import os
import random
import dill
from tqdm import tqdm
from func_timeout import func_timeout, FunctionTimedOut
import logging
import traceback

from logging import getLogger

from dataset import WebNLG, DataEntry, OpenDialKGR
import dataset as ds
import ollama
from ollama import Client
from pydantic import BaseModel
from typing import List, Dict, Any, Tuple

NUM_CTX=20000

logger = getLogger('reflection_trainer')


class RDFTriple(BaseModel):
    subject: str
    predicate: str
    object: str
class Example(BaseModel):
    input: list[RDFTriple]
    output: str
class Dataset(BaseModel):
    examples: list[Example]

    
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
    
    
class TestEngineer:
    TEMPLATE = '''
Your task is to generate a dataset for data-to-text task. More precisely, for converting RDF triples into plain text. Each example in the dataset should contain: input (a set of RDF triples) and output (verbalization). For instance:		

Input: [RDFTriple(subject='Pontiac Rageous', predicate='production start year', object='1997'), RDFTriple(subject='Pontiac Rageous', predicate='assembly', object='Michigan'), RDFTriple(subject='Pontiac Rageous', predicate='production end year', object='1997')]
Output: 'Pontiac Rageous was first made in Michigan in 1997 and was last produced in 1997.'
	
In the generated dataset, possible `predicate` values of RDF triple are: {predicates}.

Below you have an example of RDF triple for every predicate.
{examples}

You can use RDF triples from examples above, but it is expected that you will generate new triples to construct new examples for the dataset. Note that the input may contain a single triple or multiple triples.

Generate {examples_per_request} diverse examples, each containing: input (a set of RDF triples) and output (verbalization).
    '''
    
    TEMPLATE_SINGLETON = '''
Your task is to convert an RDF triple into plain text. Your response should contain two elements: input (the triple) and output (verbalization). 

# Example

Input: [RDFTriple(subject='Pontiac Rageous', predicate='production start year', object='1997')]
Output: 'The production of Pontiac Rageous started in 1997.'
	
Now, convert the following RDF triple into plain text.
Input: {example}

Format your response as JSON following the schema: {json_schema}
    '''
    def __init__(self, language_model: LanguageModel, dataset_size=50, examples_per_request = 5, examples_per_predicate=1, add_singletons=False):
        self.dataset_size = dataset_size
        self.examples_per_predicate = examples_per_predicate
        self.add_singletons = add_singletons
        self.examples_per_request = examples_per_request
        self.language_model = language_model
        self.system_prompt = (
            "You are an expert data generator. Your task is to generate a dataset for data-to-text task. "
        )
    def generate_singleton(self, triple):
        example = f"RDFTriple(subject='{triple.subject}', predicate='{triple.predicate}', object='{triple.object}')"
        TEMPLATE = self.TEMPLATE_SINGLETON.format(example=example, json_schema=Example.model_json_schema())
        
        messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": TEMPLATE }
                ]

        response = self.language_model.query(messages, format = Example.model_json_schema(), temperature=0.7)
            
        answer = Example.model_validate_json(response["message"]["content"])
        #print(answer)
        return answer
    
    
    def generate(self, dataset, category = "Film"):
        final_dataset = []
        json_schema=Dataset.model_json_schema()
        predicate_count = {}
        mot = [i for i in dataset.data if i.category == category]
        preds = set([j.predicate for i in mot for j in i.data])
        old_preds = set(preds)
        print(f"Number of predicates: {len(preds)}")
        if self.add_singletons:
            for p in tqdm(preds):
                ex = [j for i in mot for j in i.data if j.predicate == p]
                if len(ex) > 0:
                    ex = ex[0]                  
                    answer = self.generate_singleton(ex)
                    final_dataset.append(answer)
                    predicate_count[p] = 1
        iteration = 0
        while len(final_dataset) < self.dataset_size:
            #Remove from preds predicates that have predicate_count > self.examples_per_predicate
            preds = [i for i in preds if i not in predicate_count or predicate_count[i] < self.examples_per_predicate]
            print(f"Size of the dataset: {len(final_dataset)} Predicates: {len(preds)}")
            # print(f"predicate_count: {predicate_count}")
            # print(f"preds: {preds}")
            if len(preds) == 0:
                print("No more predicates to generate. Early stopping.")
                break
            #Prepare prompt
            predicates = ", ".join([f'"{p}"' for p in preds])
            examples = []   
            for p in preds:
                ex = [j for i in mot for j in i.data if j.predicate == p]
                examples.append(ex[0])
            examples = "\n".join([f"RDFTriple(subject='{i.subject}', predicate='{i.predicate}', object='{i.object}')" for i in examples])
            TEMPLATE = self.TEMPLATE.format(predicates=predicates, examples=examples, examples_per_request=self.examples_per_request)
            
            messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": TEMPLATE  }#+ f"Format your response as JSON which follows the schema:\n {json_schema}"}
                ]

            response = self.language_model.query(messages, temperature=0.8)
            answer = response["message"]["content"]
            #Reformat to JSON
            messages.append({"role": "assistant", "content": answer})
            messages.append({"role": "user", "content": f"Restructure your previous answer into a JSON format. JSON should follow the schema:\n {json_schema}"})
            response = self.language_model.query(messages, format = json_schema, temperature=0.2)
            
            answer = Dataset.model_validate_json(response["message"]["content"])
            #Filter from answer.examples inputs that conatin predicates not in preds
            print(f"Generated {len(answer.examples)} examples")
            correct_examples = [i for i in answer.examples if all(j.predicate in old_preds for j in i.input)]
            # for i in answer.examples:
            #     a = [j for j in i.input if j.predicate not in preds]
            #     if len(a) > 0:
            #         answer.examples.remove(i)
            final_dataset.extend(correct_examples)
            print(f"Correct: {len(correct_examples)} examples")
            #print(f"Size of the dataset: {len(final_dataset)} Predicates: {len(preds)}")
            
            #Build dict counting how many times each predicate is used
            for i in correct_examples:
                for j in i.input:
                    if j.predicate not in predicate_count:
                        predicate_count[j.predicate] = 1
                    else:
                        predicate_count[j.predicate] += 1
            
            print(f"End of {iteration} iteration")
            iteration += 1
            if iteration > 40:
                break
        
        return final_dataset
    



def parse_args():
    script_dir = os.path.dirname(__file__)  
    parser = ArgumentParser()
    parser.add_argument("--out-file", type=str, help="saved model", default="refl-checkpoint")
    parser.add_argument("--outname",  default="")
    parser.add_argument("--dataset",  default="webnlg")
    parser.add_argument("--log-file",  default='example-norandom.log')
    parser.add_argument("--model", type=str, help="LLM model", default="deepseek-r1:70b")
    parser.add_argument("--config", type=int, help="LLM model", default=None)
    parser.add_argument("--ip", type=str, help="LLM model", default="10.10.25.66")
    parser.add_argument("--ipeval", type=str, help="LLM model", default="localhost")
    #bool argument full, default false
    parser.add_argument("--full", action='store_true', help="Full evaluation")
    parser.add_argument("--category", type=str, help="Full evaluation")
    return parser.parse_args()


if __name__ == '__main__':
    #MAX_ITER = 30
    args = parse_args()
    logging.basicConfig(filename=args.log_file, encoding='utf-8', level=logging.INFO)
    script_path = os.path.dirname(os.path.realpath(__file__))
    
    lm = LanguageModel(model_name=args.model, log_file="lm-norandom.logs", ip=f"{args.ip}:8889")
    #dataset_size=50, examples_per_request = 5, examples_per_predicate=1, add_singletons=False
    
    print("Loading dataset")
    if args.dataset == "webnlg":
        dataset = WebNLG()
    elif args.dataset == "opendial":
        dataset = OpenDialKGR()
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    dataset.load(['test'])
    print(f"Dataset loaded. Size: {len(dataset.data)}")
    #lm_eval.use_num_ctx = True
    lm.use_num_ctx = True
    
    if args.category is not None:
        categories = [args.category]        
    else:
        categories = set([i.category for i in dataset.data])
        
        
    for category in categories:
        gen = TestEngineer(lm, examples_per_predicate = 4, add_singletons=True, dataset_size=1000, examples_per_request = 50)
        new_dataset = gen.generate(dataset, category = category)
        
        data = []
        for i, example in enumerate(new_dataset):
            triples = [ds.RDFTriple(i.subject, i.predicate, i.object) for i in example.input]
            entry = DataEntry(
                        data=triples, refs=[example.output], data_type="triples",
                        entry_id=str(i), category=category
                    )
            data.append(entry)
                    
        with open(args.out_file+"-"+category, 'wb') as f:
            dill.dump(data, f)
