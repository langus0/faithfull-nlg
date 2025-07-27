import json
import dill  
import sys
import traceback
from typing import *
from datetime import datetime
from dataset import RDFTriple
import json
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
class VLanguageModel():
    def __init__(self, model_name="casperhansen/llama-3.3-70b-instruct-awq"):
        self.llm = LLM(model=model_name,  max_model_len=2000,
                       gpu_memory_utilization=0.80, max_num_seqs=1, tensor_parallel_size=2,enable_prefix_caching=True, max_num_batched_tokens=2000) 

    def generate(self, prompts, temperature=0.7, seed=None):
        sampling_params = SamplingParams(
            temperature=temperature, max_tokens=5000)
        outputs = self.llm.generate(prompts, sampling_params)
        outputs = [output.outputs[0].text for output in outputs]
        return outputs
    def chat(self, prompts, temperature=0.8, seed=None, guided_decoding_params=None):
        sampling_params = SamplingParams(
            temperature=temperature, max_tokens=10,repetition_penalty = 1.1, top_k=40, top_p=0.9)   
        if guided_decoding_params is not None:
            sampling_params.guided_decoding = guided_decoding_params
        outputs = self.llm.chat(prompts, sampling_params)
        outputs = [output.outputs[0].text for output in outputs]
        return outputs   

class EmptyLM():

    def query(self, messages, temperature=0.1, seed=None, options=None, format=None):
        return ["0"] * len(messages)
#from program import Program, ProgramWriter
# from lm_poller import LMPoller
#from text_preprocessing import extract_triplets, extract_relations
#from lm_response_evaluator import extract_code, get_response_similarity, evaluate_response


fluency = "Fluency: Is it possible to say that the text progresses naturally, forms a coherent whole and it is easy to understand the text?"
gram = "Grammatical Correctness: The text should be free of grammatical and semantic errors."

faith = "Faithfulness: Every piece of information mentioned in the text should be verifiable/supported/inferred from the input data only. The text should be penalized if any piece of information is not verifiable/supported/inferred from the input data or if the text overgeneralizes something."

add = "Addition: Does the text contain only facts that are mentioned in the data? Not all facts need to be mentioned, but any that are should be supported by data."

omm = "Omissions: Does the text include descriptions of all predicates presented in the data? While facts not supported by the data can be mentioned, all facts contained within the input data must be included."



import os
files = [f for f in os.listdir('../') if f.startswith(sys.argv[1]) and not f.endswith('full')]

print("----")
print(files)
base_prefix = sys.argv[2] + "-"
cat2prog = {}
for file in files:
    category = file.split('-')[-1]
    with open("../"+file, 'rb') as f:
                    program = dill.load(f)
                    curr_iteration = dill.load(f) 
                    best_program = dill.load(f)
                    best_num_successes_test = dill.load(f)  
    if category in cat2prog:
        curr_best = cat2prog[category]
        if best_num_successes_test > curr_best[1]:
            cat2prog[category] = (file, best_num_successes_test)
            print(f"{category} Replacing with {file}")
    else:
        cat2prog[category] = (file, best_num_successes_test)
        print(f"{category} Replacing with {file}")
cat2file = {k: v[0] for k, v in cat2prog.items()}

category2program = {}
for cat, file in cat2file.items():
    with open("../"+file, 'rb') as f:
                    program = dill.load(f)
                    #print(program)
                    curr_iteration = dill.load(f) 
                    best_program = dill.load(f)
                    best_num_successes_test = dill.load(f)
                    # reflection = dill.load(f)
                    # feedback = dill.load(f)
                    # is_new_idea_needed = dill.load(f)
                    # idea = dill.load(f)
                    # program_struct = dill.load(f)
                    # errors = dill.load(f)
    print(f"Loaded {cat} program with {best_num_successes_test} successes obtained in {curr_iteration} iterations.")
    category2program[cat] = best_program


#=========================================
from func_timeout import func_timeout, FunctionTimedOut
class Evaluator:
    CODE = '''
from collections import defaultdict, namedtuple

{program}

RDFTriple = namedtuple("RDFTriple", ["subject", "predicate", "object"])
triples = {triples}
output = verbalize_set_of_triples(triples)
result_dict['output'] = output
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
    def __init__(self, log_file=None):
        pass
    def execute_program(self, program, triplets):
        try:
            return func_timeout(5, Evaluator.execute_program2, args=(self, program, triplets), kwargs=None)
        except FunctionTimedOut:
            # Handle exceptions
            return '', "Error: Program did not terminate within 5 seconds."   
        
    def execute_program2(self, program, sample):
        result_dict = {}
        combined_script = self.CODE.format(triples = sample.data, program=program)

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
            return output, "ERROR:   " + err






from evaluate_program import MultiReferenceMetric, ReferenceLessMetric, get_basic_metrics, BERTScore, BLEURT, WebNLG, LLMJudgeGrammaticality, LLMJudgeOmmisions, LLMJudgeAdditions, LLMJudgeFluency, WebNLGgemCFA, WebNLGgemFI, WebNLGgemFA, LLMJudgeFaithfulness, ThemisExtractor, OpenDialKGR
#from reflection_trainer import Evaluator
#from idea_trainer import LanguageModel, EmptyLM
import numpy as np
def evaluate_outs(category2program, metrics, dataset_const=WebNLG):
    data = dataset_const()
    data.load(['test'])
    eval = Evaluator(None)

    ood_categories = []
    refs_single = []
    preds_single = []
    refs_multi = []
    preds_multi = []
    input_multi = []
    ref_lens = []
    is_out_domain = []
    for i, dataEntry in enumerate(data.data):
        is_out_domain.append(dataEntry.category in ["Film", "MusicalWork", "Scientist", "all"])

        #relations = tuple(sorted([i.pred for i in dataEntry.data]))
        # input = [tuple([triplet.subj, triplet.pred, triplet.obj]) for triplet in dataEntry.data]
        if dataEntry.category in category2program:
            output, err = eval.execute_program(category2program[dataEntry.category],dataEntry)
            if err is not None:
                output=  "error"
                print(f"ERROR {dataEntry.category}")
        else: 
            output = "OUT OF DOMAIN"
            ood_categories.append(dataEntry.category)
            is_out_domain[-1] = True

        refs_multi.append(dataEntry.refs)
        preds_multi.append(output)
        input_multi.append(dataEntry.data)
        
        for reference_text in dataEntry.refs:
            refs_single.append(reference_text)
            preds_single.append(output)
        ref_lens.append(len(dataEntry.refs))
    print(f"Out of domain categories: {set(ood_categories)}")
    is_out_domain = np.array(is_out_domain)

    for metric in metrics:
        if isinstance(metric, MultiReferenceMetric):
            metric.compute(preds_multi, refs_multi, ref_lens, is_out_domain)
        elif isinstance(metric, ReferenceLessMetric):
            metric.compute(preds_multi, input_multi, ref_lens, is_out_domain)
        else:
            metric.compute(preds_single, refs_single, ref_lens, is_out_domain)
    return preds_multi, refs_multi



metrics = []
metrics.extend(get_basic_metrics())
metrics.append(BERTScore())
metrics.append(BLEURT())

preds, ref = evaluate_outs(category2program, metrics, dataset_const = OpenDialKGR)

lm = VLanguageModel() #LanguageModel(log_file=None, ip=f"{sys.argv[3]}:8889")
#lm = EmptyLM()
#lm.use_num_ctx = True
#metrics.extend([LLMJudgeGrammaticality(lm, binary=True), LLMJudgeOmmisions(lm, binary=True), LLMJudgeAdditions(lm, binary=True), LLMJudgeFluency(lm, binary=True)])

prefix = base_prefix + "STD"
metrics= []

metrics.extend([LLMJudgeGrammaticality(lm, binary=True), LLMJudgeOmmisions(lm, binary=True), LLMJudgeAdditions(lm, binary=True)])


print("WEBNLG")
preds, ref = evaluate_outs(category2program, metrics, dataset_const = OpenDialKGR)

prefix = base_prefix + "CFA"
metrics= []

metrics.extend([LLMJudgeGrammaticality(lm, binary=True), LLMJudgeOmmisions(lm, binary=True), LLMJudgeAdditions(lm, binary=True)])


print("WEBNLG==CFA")
preds, ref = evaluate_outs(category2program, metrics, dataset_const = WebNLGgemCFA)

prefix = base_prefix + "FI"
metrics= []

metrics.extend([LLMJudgeGrammaticality(lm, binary=True), LLMJudgeOmmisions(lm, binary=True), LLMJudgeAdditions(lm, binary=True)])


print("WEBNLG==FI")
preds, ref = evaluate_outs(category2program, metrics, dataset_const = WebNLGgemFI)

prefix = base_prefix + "FA"
metrics= []

metrics.extend([LLMJudgeGrammaticality(lm, binary=True), LLMJudgeOmmisions(lm, binary=True), LLMJudgeAdditions(lm, binary=True)])


print("WEBNLG==FA")
preds, ref = evaluate_outs(category2program, metrics, dataset_const = WebNLGgemFA)

