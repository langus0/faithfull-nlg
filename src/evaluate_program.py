import logging
import evaluate
import numpy as np

from collections import defaultdict, namedtuple
from datasets import load_dataset
from text_preprocessing import normalize


logger = logging.getLogger(__name__)

RDFTriple2 = namedtuple("RDFTriple", ["subj", "pred", "obj"])
RDFTriple = namedtuple("RDFTriple", ["subject", "predicate", "object"])


class DataEntry:
    """
    An entry in the dataset
    """

    def __init__(self, data, refs, data_type, entry_id, align=None, num_ref_sentences=None, category=None, dialhist=None):
        self.data = data
        self.refs = refs
        self.data_type = data_type
        self.align = align
        self.num_ref_sentences = num_ref_sentences
        self.category = category
        self.dialhist = dialhist
        self.entry_id = entry_id.replace("/", "_")

    def __repr__(self):
        return str(self.__dict__)


class OpenDialKGR:

    name = "OpenDialKGR"

    def __init__(self, *args, **kwargs):
        self.data = []
        
    def normalize_preds(self, x):
        
        if x.predicate.startswith("~"):
            return RDFTriple(x.object, x.predicate[1:], x.subject)
        return x

    def load(self, splits, path="data/OpenDialKGR"):
        import json
        import os
        # load the dataset from HF datasets
        split = splits[0]
        with open(os.path.join(path,split+".json")) as f:
            lines = f.read()
        examples = json.loads(lines)

        for i, example in enumerate(examples["data"]):
                triples = example["in"]
                
                triples = [t.split("|") for t in triples.split('▸')]
                triples = [(normalize(x, remove_parentheses=False) for x in t) for t in triples]
                triples = [RDFTriple(*t) for t in triples]
                triples = [self.normalize_preds(t) for t in triples]

                refs = [example["out"]]
                entry = DataEntry(data=triples, refs=refs, data_type="triples", category="all", entry_id=str(i))
                self.data.append(entry)
                
class WebNLG:
    """
    The WebNLG dataset: https://gem-benchmark.com/data_cards/web_nlg
    Contains RDF triples from DBPedia and their crowdsourced verbalizations.
    """

    name = "webnlg"

    def __init__(self, *args, **kwargs):
        self.data = []

    def load(self, splits, path=None):
        # load the dataset from HF datasets
        dataset = load_dataset("gem", "web_nlg_en")

        for split in splits:
            data = dataset[split if split != "dev" else "validation"]

            for example in data:
                triples = example["input"]
                triples = [t.split("|") for t in triples]
                triples = [(normalize(x, remove_parentheses=False) for x in t) for t in triples]
                triples = [RDFTriple(*t) for t in triples]

                if split == "test":
                    refs = example["references"]
                else:
                    refs = [example["target"]]

                entry = DataEntry(
                    data=triples, refs=refs, data_type="triples",
                    entry_id=example['webnlg_id'], category=example["category"]
                )
                self.data.append(entry)
class WebNLGgem:
    name = "webnlggem"

    def __init__(self, *args, **kwargs):
        self.data = []
        
    def parse_webnlg_xml(self, xml_string):
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml_string)
        entries = []

        for entry in root.findall(".//entry"):
            eid = entry.get("eid")
            category = entry.get("category")

            # Get modified triples as strings
            modified_triples = [
                mtriple.text.strip()
                for mtriple in entry.find("modifiedtripleset").findall("mtriple")
            ]

            entry_dict = {
                "webnlg_id": eid,
                "input": modified_triples,
                "category": category
            }
            entries.append(entry_dict)

        return entries
    def load(self, splits, path):
        # load the dataset from HF datasets
        with open(path, 'r', encoding='utf-8') as file:
            xml_string = file.read()
        dataset = self.parse_webnlg_xml(xml_string)

        for example in dataset:
                triples = example["input"]
                triples = [t.split("|") for t in triples]
                triples = [(normalize(x, remove_parentheses=False) for x in t) for t in triples]
                triples = [RDFTriple(*t) for t in triples]

                refs = [""]

                entry = DataEntry(data=triples, entry_id=example["webnlg_id"], refs=refs, data_type="triples", category=example["category"])
                self.data.append(entry)

class WebNLGgemCFA(WebNLGgem):
    name = "WebNLGgemCFA"
    def load(self, splits):
        path="GEM-v2-D2T-SharedTask/D2T-1-CFA_WebNLG_CounterFactual.xml"
        super().load(None, path)
class WebNLGgemFA(WebNLGgem):
    name = "WebNLGgemFA"
    def load(self, splits):
        path="GEM-v2-D2T-SharedTask/D2T-1-FA_WebNLG_Factual.xml"
        super().load(None, path)        
class WebNLGgemFI(WebNLGgem):
    name = "WebNLGgemFI"
    def load(self, splits):
        path="GEM-v2-D2T-SharedTask/D2T-1-FI_WebNLG_Fictional.xml"
        super().load(None, path)
       
       
class AugmentedDataset:
    def __init__(self, *args, **kwargs):
        self.data = []

    def load(self, path):
        def get_known_relations():
            from evaluate_program import WebNLG
            data = WebNLG()
            data.load(['test', "train"])

            triplets = [dataEntry.data for dataEntry in data.data]
            triplets = [t.pred  for sets in triplets for t in sets]
            known_relations = set(triplets)
            return known_relations
        
        def convert2triple(input, known_relations):
            if len(input) == 1:
                input = input[0].split(",")
            input = [i.strip() for i in input]
            rel = [i for i,j in enumerate(input) if j in known_relations]
            if len(rel) != 1:
                print("ERROR!")
                print(input)
                print(rel)
                return None
            i = rel[0]
            return [", ".join(input[:i]), input[i], ", ".join(input[i+1:])]
            
        import json
        # load the dataset from HF datasets
        with open(path, "r") as f:
            data = json.load(f)

        known_relations = get_known_relations()
        for id, example in  enumerate(data["not_augmented_samples"]):
            triples = example["in"]
            # print(id)
            import re
            triples = re.findall(r'\(.*?\)', triples)
            triples = [i[1:-1] for i in triples]
            # print(triples)
            triples = [convert2triple(t.split("|"), known_relations) for t in triples]
            if any([t is None for t in triples]):
                print(triples)
                print(f"SKIP {id}")
                continue
            # print(triples)
            triples = [(normalize(x, remove_parentheses=False) for x in t) for t in triples]
            
            triples = [RDFTriple(*t) for t in triples]
            if "out" not in example:
                print(f"WARN: Incomplete example {example}")
                continue
            entry = DataEntry(
                data=triples, refs=[example["out"]], data_type="triples",
                entry_id=str(id), category="augmented"
            )
            self.data.append(entry)



# METRICS ==============================

class SingleReferenceMetric:
    def __init__(self) -> None:
        self.name = "SingleReferenceMetric"

    def eval(self, preds, refs, ref_lens, is_out_domain):
        pass

    def compute(self, preds, refs, ref_lens, is_out_domain):
        results = self.eval(preds, refs, ref_lens, is_out_domain)

        i = 0
        merged_results = []
        for len_r in ref_lens:
            merged_results.append(results[i:i + len_r].mean())
            i += len_r

        results = np.array(merged_results)
        print(
            f"{self.name}: {results.mean()} +- {results.std()}; OOD: {results[is_out_domain].mean()}; InD: {results[~is_out_domain].mean()}")

class ReferenceLessMetric:
    def __init__(self) -> None:
        self.name = "ReferenceLessMetric"

    def eval(self, preds, inputs, ref_lens, is_out_domain):
        pass

    def compute(self, preds, inputs, ref_lens, is_out_domain):
        results = self.eval(preds, inputs, ref_lens, is_out_domain)

        results = np.array(results)
        print(
            f"{self.name}: {results.mean()} +- {results.std()}; OOD: {results[is_out_domain].mean()}; InD: {results[~is_out_domain].mean()}")


class ThemisExtractor(ReferenceLessMetric):
    def __init__(self, aspect, out_file) -> None:
        self.name = "ThemisExtractor"
        self.template = {"target_des":"Text", "source_des":"Data", "task":"Data to text", "aspect": aspect}
        self.out_file = out_file
        
    def process_input(self, inpu, output):
        mydict = self.template.copy()
        sample = [f"({s.subject}, {s.predicate}, {s.object})" for s in inpu]
        mydict["source"] = ", ".join(sample)
        mydict["target"] = output
        return mydict
        

    def eval(self, preds, inputs, ref_lens, is_out_domain):
        result = []
        for inpu, output in zip(inputs,preds):
            result.append( self.process_input(inpu, output)   ) 
        #save result as JSON to the out_file
        import json
        with open(self.out_file, "w") as f:
            json.dump(result, f, indent=4)
        print(f"{self.name} saved results to {self.out_file}")
        return np.array([0]*len(inputs))

    def compute(self, preds, inputs, ref_lens, is_out_domain):
        results = self.eval(preds, inputs, ref_lens, is_out_domain)

        results = np.array(results)
        print(
            f"{self.name}: {results.mean()} +- {results.std()}; OOD: {results[is_out_domain].mean()}; InD: {results[~is_out_domain].mean()}")


class MultiReferenceMetric:
    def __init__(self) -> None:
        self.name = "MultiReferenceMetric"

    def eval(self, preds, refs, ref_lens, is_out_domain):
        pass

    def compute(self, preds, refs, ref_lens, is_out_domain):
        results = self.eval(preds, refs, ref_lens, is_out_domain)
        print(f"{self.name}: {results} ")
        return results


class BLEURT(SingleReferenceMetric):
    def __init__(self) -> None:
        super().__init__()
        self.metric = evaluate.load("bleurt", module_type="metric")
        self.name = "BLEURT"

    def eval(self, preds, refs, ref_lens, is_out_domain):
        results = self.metric.compute(predictions=preds, references=refs)
        # results_in = self.metric.compute(predictions=[p for i,p in enumerate(preds) if not is_out_domain[i]], references=[r for i,r in enumerate(refs) if not is_out_domain[i]])
        return np.array(results["scores"])


class BERTScore(SingleReferenceMetric):
    def __init__(self) -> None:
        super().__init__()
        self.metric = evaluate.load("bertscore")
        self.name = "BERTScore"

    def eval(self, preds, refs, ref_lens, is_out_domain):
        results = self.metric.compute(predictions=preds, references=refs, lang="en")
        # results_in = self.metric.compute(predictions=[p for i,p in enumerate(preds) if not is_out_domain[i]], references=[r for i,r in enumerate(refs) if not is_out_domain[i]], lang="en")
        return np.array(results["f1"])

class LLMJudge(ReferenceLessMetric):
    def __init__(self, lm) -> None:
        super().__init__()
        self.language_model = lm
        self.name = "LLMJudge"

    def eval(self, preds, inputs, ref_lens, is_out_domain):
        from tqdm import tqdm
        prompts = []
        for inpu, output in tqdm(zip(inputs,preds), total=len(inputs)):
            prompts.append(self.check_output(inpu, output) )
        outputs = self.language_model.chat(prompts, temperature= 0.0)
        scores = []
        for out in outputs:
            answer =out.strip().lower().split()[0]
            if not answer.isdigit():
                print(f"LLMJudge: answer is not a number: {answer}\nResponse: {out}\n===\n")
                scores.append(0)
            else:
                scores.append(int(answer))
        return np.array(scores)

    def check_output(self, sample, output):
        sample = [f"({s.subject}, {s.predicate}, {s.object})" for s in sample]
        prompt = self.format_prompt(", ".join(sample), output)
        messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]

     
        return messages
    
    def format_prompt(self, sample, output):
        pass
        
class LLMJudgeO(ReferenceLessMetric):
    def __init__(self, lm) -> None:
        super().__init__()
        self.language_model = lm
        self.name = "LLMJudge"

    def eval(self, preds, inputs, ref_lens, is_out_domain):
        from tqdm import tqdm
        scores = []
        for inpu, output in tqdm(zip(inputs,preds), total=len(inputs)):
            score = self.check_output(inpu, output)    
            scores.append(score)
        return np.array(scores)

    def check_output(self, sample, output):
        sample = [f"({s.subject}, {s.predicate}, {s.object})" for s in sample]
        prompt = self.format_prompt(", ".join(sample), output)
        messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]

        options = {
                   "num_predict": 5
                       }
        response = self.language_model.query(messages, options=options, temperature=0.0)
        answer = response["message"]["content"].strip().lower().split()[0]
        #check if answer is a number
        if not answer.isdigit():
            print(f"LLMJudge: answer is not a number: {answer}\nPrompt: {prompt}\nResponse: {response['message']['content']}\n===\n")
            return 0

        return int(answer)
    
    def format_prompt(self, sample, output):
        pass
    
class LLMJudgeGrammaticality(LLMJudge):
    def __init__(self, lm, binary=False) -> None:
        super().__init__(lm)
        self.name = "LLMJudge Grammar"
        self.system_prompt = "You are an expert evaluator of data-to-text generation task."
        self.binary = binary

    def format_prompt(self, sample, output):
        if not self.binary:
            return f"""Your task is to evaluate the output of a data-to-text task, for which the model was instructed to produce a verbalisation of a given set of RDF triples. 
    
    You should assess **the grammatical correctness** of the resulting text. Do not take any other factors into account. Do not make assumptions or consider external knowledge not present in the provided context. Identify only errors relating to the grammaticality of the text. Do not consider aspects such as fluency, omissions or hallucinations.

Respond using a Likert scale from 1 to 5, where 1 is the worst score and 5 is the best.

System output: {output}

Assess the grammatical correctness of the output. Answer with a single number from 1 (worst) to 5 (best), without any other text.
"""
        else:
            return f"""Your task is to evaluate the output of a data-to-text task, for which the model was instructed to produce a verbalisation of a given set of RDF triples. 
    
    You should assess **the grammatical correctness** of the resulting text. Do not take any other factors into account. Do not make assumptions or consider external knowledge not present in the provided context. Identify only errors relating to the grammaticality of the text. Do not consider aspects such as fluency, omissions or hallucinations.

Respond with 1 for correct and 0 for incorrect.

System output: {output}

Assess the grammatical correctness of the output. Answer with a single number 1 (correct) or 0 (incorrect), without any other text.
"""

class LLMJudgeFaithfulness(LLMJudge):
    def __init__(self, lm, binary=False) -> None:
        super().__init__(lm)
        self.name = "LLMJudge faithfulness"
        self.system_prompt = "You are an expert evaluator of data-to-text generation task."
        self.binary = binary

    def format_prompt(self, sample, output):
        if not self.binary:
            return f"""Your task is to evaluate the output of a data-to-text task, for which the model was instructed to produce a verbalisation of a given set of RDF triples. 
    
You should assess the **faithfulness** of the resulting text, i.e. whether all the information in the input triples has been faithfully conveyed in the output. A faithful output text should include all the facts expressed in the input triples without any new information being added or any existing information being distorted.

Do not take any other factors into account. Do not make assumptions or consider external knowledge not present in the provided context. Identify only errors relating to the fluency of the text. Do not consider aspects such as grammaticality or fluency.

Respond using a Likert scale from 1 to 5, where 1 is the worst score and 5 is the best.

Input triples: {sample}
System output: {output}

Assess the faithfulness of the output. Answer with a single number from 1 (whole output is hallucinated) to 5 (faithfull). Output just a single number, without any other text.
"""
        else:
            return f"""Your task is to evaluate the output of a data-to-text task, for which the model was instructed to produce a verbalisation of a given set of RDF triples. 
    
You should assess the **faithfulness** of the resulting text, i.e. whether all the information in the input triples has been faithfully conveyed in the output. A faithful output text should include all the facts expressed in the input triples without any new information being added or any existing information being distorted.

 Do not take any other factors into account. Do not make assumptions or consider external knowledge not present in the provided context. Identify only errors relating to the fluency of the text. Do not consider aspects such as grammaticality, omissions or hallucinations.

Respond with 1 if text is faithfull and 0 if the text contains hallucinations.

Input triples: {sample}
System output: {output}

Assess the faithfulness of the output. Answer with a single number: 1 (faithfull) or 0 (hallucinations), without any other text.
"""

class LLMJudgeFluency(LLMJudge):
    def __init__(self, lm, binary=False) -> None:
        super().__init__(lm)
        self.name = "LLMJudge Fluency"
        self.system_prompt = "You are an expert evaluator of data-to-text generation task."
        self.binary = binary

    def format_prompt(self, sample, output):
        if not self.binary:
            return f"""Your task is to evaluate the output of a data-to-text task, for which the model was instructed to produce a verbalisation of a given set of RDF triples. 
    
You should assess the **fluency** of the resulting text, i.e. whether it reads well. Do not take any other factors into account. Do not make assumptions or consider external knowledge not present in the provided context. Identify only errors relating to the fluency of the text. Do not consider aspects such as grammaticality, omissions or hallucinations.

Respond using a Likert scale from 1 to 5, where 1 is the worst score and 5 is the best.

System output: {output}

Assess the fluency of the output. Answer with a single number: 1 (fluent) or 5 (not fluent), without any other text.
"""
        else:
            return f"""Your task is to evaluate the output of a data-to-text task, for which the model was instructed to produce a verbalisation of a given set of RDF triples. 
    
You should assess the **fluency** of the resulting text, i.e. whether it reads well. Do not take any other factors into account. Do not make assumptions or consider external knowledge not present in the provided context. Identify only errors relating to the fluency of the text. Do not consider aspects such as grammaticality, omissions or hallucinations.

Respond with 1 if text is fluent and 0 if the text is not fluent.

System output: {output}

Assess the fluency of the output. Answer with a single number: 1 (fluent) or 0 (not fluent), without any other text.
"""
class LLMJudgeOmmisions(LLMJudge):
    def __init__(self, lm, binary=False) -> None:
        super().__init__(lm)
        self.name = "LLMJudge Ommisions"
        self.system_prompt = "You are an expert evaluator of data-to-text generation task."
        self.binary = binary

    def format_prompt(self, sample, output):
        if not self.binary:
            return f"""Your task is to evaluate the output of a data-to-text task, for which the model was instructed to produce a verbalisation of a given set of RDF triples. 
    
You should assess the **omissions** in the resulting text; in other words, you should check whether any of the input triples were not verbalised.  You can perform the task by iterating over the input triples and checking if it is present in the output. Do not take any other factors into account. Do not make assumptions or consider external knowledge not present in the provided context. Identify only errors relating to the fluency of the text. Do not consider aspects such as grammaticality, fluency or the addition of new facts (hallucinations).

Respond using a Likert scale from 1 to 5, where 1 is the worst score and 5 is the best.

Input triples: {sample}
System output: {output}

Assess the omissions of the input triples. Answer with a single number from 1 (worst) to 5 (best), without any other text.
"""
        else:
            return f"""Your task is to evaluate the output of a data-to-text task, for which the model was instructed to produce a verbalisation of a given set of RDF triples. 
    
You should assess the **omissions** in the resulting text; in other words, you should check whether any of the input triples were not verbalised.  You can perform the task by iterating over the input triples and checking if it is present in the output. Do not take any other factors into account. Do not make assumptions or consider external knowledge not present in the provided context. Identify only errors relating to the fluency of the text. Do not consider aspects such as grammaticality, fluency or the addition of new facts (hallucinations).

Respond with 1 if any of the input triples is ommited and 0 if not.

Input triples: {sample}
System output: {output}

Assess the omissions of the input triples. Answer with a single number: 1 (omissions) to 0 (no omissions), without any other text.
"""
    
class LLMJudgeAdditions(LLMJudge):
    def __init__(self, lm, binary=False) -> None:
        super().__init__(lm)
        self.system_prompt = "You are an expert evaluator of data-to-text generation task."
        self.name = "LLMJudgeAdditions"
        self.binary = binary

    def format_prompt(self, sample, output):
        if not self.binary:
            return f"""Your task is to evaluate the output of a data-to-text task, for which the model was instructed to produce a verbalisation of a given set of RDF triples. 
    
You should assess the **addition of new facts** in the resulting text which were not present in the input. You can perform the task by carefully reading the text and checking if the facts mentioned are present in the input triples. Do not take any other factors into account. Do not make assumptions or consider external knowledge not present in the provided context. Identify only errors relating to the fluency of the text. Do not consider aspects such as grammaticality, fluency or the omissions of input triples.

Respond using a Likert scale from 1 to 5, where 1 is the worst score and 5 is the best.

Input triples: {sample}
System output: {output}

Assess the additions of new facts in the output. Answer with a single number from 1 (worst) to 5 (best), without any other text.
"""
        else:
            return f"""Your task is to evaluate the output of a data-to-text task, for which the model was instructed to produce a verbalisation of a given set of RDF triples. 
    
You should assess the **addition of new facts** in the resulting text which were not present in the input. You can perform the task by carefully reading the text and checking if the facts mentioned are present in the input triples. Do not take any other factors into account. Do not make assumptions or consider external knowledge not present in the provided context. Identify only errors relating to the fluency of the text. Do not consider aspects such as grammaticality, fluency or the omissions of input triples.

Respond with 1 if the output contains facts not mentioned in the input and 0 if not.

Input triples: {sample}
System output: {output}

Assess the additions of new facts in the output. Answer with a single number: 1 (additions) or 0 (no additions), without any other text.
"""
    
class BLEU(MultiReferenceMetric):
    def __init__(self) -> None:
        super().__init__()
        self.metric = evaluate.load("bleu")
        self.name = "BLEU"

    def eval(self, preds, refs, ref_lens, is_out_domain):
        results = self.metric.compute(predictions=preds, references=refs)        
        #results_in = self.metric.compute(predictions=[p for i,p in enumerate(preds) if not is_out_domain[i]], references=[r for i,r in enumerate(refs) if not is_out_domain[i]])
        results_out = self.metric.compute(predictions=[p for i,p in enumerate(preds) if  is_out_domain[i]], references=[r for i,r in enumerate(refs) if  is_out_domain[i]])
        good = [(i,j) for i,j in zip(preds,refs) if i not in ("SPLIT NEEDED", "OUT OF DOMAIN")]
        results_good = self.metric.compute(predictions=[i for i,j in good], references=[j for i,j in good])
        return np.array(results["bleu"]), "in", 0,"out", np.array(results_out["bleu"]),  np.array(results_good["bleu"]), len(good)/sum(is_out_domain)
    
class METEOR(MultiReferenceMetric):
    def __init__(self) -> None:
        super().__init__()
        self.metric = evaluate.load("meteor")
        self.name = "METEOR"

    def eval(self, preds, refs, ref_lens, is_out_domain):
        results = self.metric.compute(predictions=preds, references=refs)
        #results_in = self.metric.compute(predictions=[p for i,p in enumerate(preds) if not is_out_domain[i]], references=[r for i,r in enumerate(refs) if not is_out_domain[i]])
        results_out = self.metric.compute(predictions=[p for i,p in enumerate(preds) if  is_out_domain[i]], references=[r for i,r in enumerate(refs) if  is_out_domain[i]])
        #results_in = self.metric.compute(predictions=preds[~is_out_domain], references=refs[~is_out_domain])
        return np.array(results["meteor"]), "in", 0, "out", np.array(results_out["meteor"])


# EVAL ==============================

def get_basic_metrics():
    return [METEOR(), BLEU()]

from tqdm import tqdm

def evaluate_program(program, metrics):
    data = WebNLG()
    data.load(['test'])

    refs_single = []
    preds_single = []
    refs_multi = []
    preds_multi = []
    ref_lens = []
    is_out_domain = []
    for dataEntry in tqdm(data.data):
        is_out_domain.append(dataEntry.category in ["Film", "MusicalWork", "Scientist"])

        relations = tuple(sorted([i.pred for i in dataEntry.data]))
        # input = [tuple([triplet.subj, triplet.pred, triplet.obj]) for triplet in dataEntry.data]
        output = program.process_input(relations, dataEntry.data)
        if output == "OUT OF DOMAIN":
            is_out_domain[-1] = True

        refs_multi.append(dataEntry.refs)
        preds_multi.append(output)
        for reference_text in dataEntry.refs:
            refs_single.append(reference_text)
            preds_single.append(output)
        ref_lens.append(len(dataEntry.refs))
    is_out_domain = np.array(is_out_domain)

    for metric in metrics:
        if isinstance(metric, MultiReferenceMetric):
            metric.compute(preds_multi, refs_multi, ref_lens, is_out_domain)
        else:
            metric.compute(preds_single, refs_single, ref_lens, is_out_domain)
    return preds_multi, refs_multi


