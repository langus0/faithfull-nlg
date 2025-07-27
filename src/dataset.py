import json
import random
from pathlib import Path
from typing import List, Dict, Any




# class WebNLGDataset(Dataset):
#     def __init__(self, data_dir: str, split: str, samples_per_relation_set: int):
#         self.__data_dir = data_dir
#         self.__split = split
#         self.__data = self.load_data()
#         self.__data_keys = list(self.__data.keys())
#         self.__samples_per_relation_set = samples_per_relation_set

#     def load_data(self):
#         train_path = Path(self.__data_dir) / 'train.json'
#         val_path = Path(self.__data_dir) / 'dev.json'
#         test_path = Path(self.__data_dir) / 'test.json'

#         if self.__split == 'train':
#             data_path = train_path
#         elif self.__split == 'val':
#             data_path = val_path
#         elif self.__split == 'test':
#             data_path = test_path
#         else:
#             raise ValueError('Invalid split name. Use train, val, or test')

#         with open(data_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)

#         return preprocess_dataset_file(data)

#     def __len__(self):
#         return len(self.__data)

#     def __getitem__(self, idx: int | slice) -> tuple[tuple[str, str, str], list[dict[str, str]]] | tuple[tuple, Any]:
#         if isinstance(idx, int):  # Single index
#             key = self.__data_keys[idx]
#             samples = self.__data[key]

#             if self.__samples_per_relation_set >= len(samples):
#                 return key, samples

#             random_samples = random.sample(samples, self.__samples_per_relation_set)
#             return key, random_samples

#         elif isinstance(idx, slice):  # Slice
#             start, stop, step = idx.indices(len(self.__data_keys))
#             sliced_keys = self.__data_keys[idx]
#             sliced_samples = [self.__data[key] for key in sliced_keys]

#             # If step is 1, return list of tuples (keys, samples) for each index
#             if step == 1:
#                 return [(key, self.__data[key]) for key in sliced_keys]

#             # Otherwise, return a single tuple (keys, samples)
#             return sliced_keys, sliced_samples

# def preprocess_dataset_file(json_data: Dict[str, List[Dict[str, str]]]) -> dict[tuple, list[dict[str, str]]]:
#     data_dict = {}
#     for i in range(len(json_data['data'])):
#         sample = json_data['data'][i]
#         in_data = json_data['data'][i]['in']
#         relations = extract_relations(in_data)
#         relations = tuple(relations)

#         if relations in data_dict:
#             data_dict[relations].append(sample)
#         else:
#             data_dict[relations] = [sample]

#     return data_dict


import logging
import numpy as np

from collections import defaultdict, namedtuple
from datasets import load_dataset
from text_preprocessing import normalize, extract_triplets


logger = logging.getLogger(__name__)

RDFTriple = namedtuple("RDFTriple", ["subject", "predicate", "object"])

# class RDFTriple:
#     def __init__(self, subject, predicate, object):
#         self.subject = subject
#         self.predicate = predicate
#         self.object = object
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

                entry = DataEntry(data=triples, refs=refs, data_type="triples", category=example["category"])
                self.data.append(entry)


class WebNLGgemCFA(WebNLGgem):
    def load(self):
        path="GEM-v2-D2T-SharedTask/D2T-1-CFA_WebNLG_CounterFactual.xml"
        super().load(None, path)
class WebNLGgemFA(WebNLGgem):
    def load(self):
        path="GEM-v2-D2T-SharedTask/D2T-1-FA_WebNLG_Factual.xml"
        super().load(None, path)        
class WebNLGgemFI(WebNLGgem):
    def load(self):
        path="GEM-v2-D2T-SharedTask/D2T-1-FI_WebNLG_Fictional.xml"
        super().load(None, path)
       
       

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
