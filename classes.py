from config import MAX_LEN_PROMPT, MAX_LEN_OPT
from utils import encode, clean_text
import json
import onnxruntime as ort
from typing import Dict, List
import numpy as np

# Query class to hold request data
class Query:
    def __init__(self, prompt: str, options: Dict, prediction_count: int, option_text: bool):
        self.prompt = prompt
        self.options = options
        self.prediction_count = prediction_count
        self.option_text = option_text

    def __str__(self) -> str:
        out = f"""
        prompt: {self.prompt}
        options: {self.options}
        prediction_count: {self.prediction_count}
        option_text: {self.option_text}
        """
        return out

# Model class to store onnx session and predict
class Model:
    def __init__(self, model_path: str, vocab_path: str):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        with open(vocab_path, 'r') as file:
            self.vocab = json.load(file)
        self.option = ["A", "B", "C", "D", "E"]

    def predict(self, query: Query) -> List[str]:
        # Prompt encoding
        p_ids, p_len = encode(clean_text(query.prompt), self.vocab, MAX_LEN_PROMPT)
        # Options encoding
        opt_ids, opt_lens = [], []
        for opt in self.option:
            ids, length = encode(clean_text(query.options[opt]), self.vocab, MAX_LEN_OPT)
            opt_ids.append(ids)
            opt_lens.append(length)
        # Retruning items
        item = {
            "prompt_ids": np.array([p_ids]),
            "prompt_len": np.array([p_len]),
            "opt_ids": np.array([opt_ids]),
            "opt_lens": np.array([opt_lens]),
            }
        # Get output
        raw_outputs = self.session.run(None, item)
        output = raw_outputs[0]
        if isinstance(output, np.ndarray):
            ranked = np.argsort(output) 
            ans = [self.option[i] for i in ranked[0]][::-1]
            return ans
        return []
