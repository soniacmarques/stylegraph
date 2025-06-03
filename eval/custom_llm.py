import json
from pydantic import BaseModel
import torch
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from deepeval.models import DeepEvalBaseLLM
from huggingface_hub import login
import os

login(token=os.environ.get("HUGGINGFACE_TOKEN"))

class CustomLlama(DeepEvalBaseLLM):
    def __init__(self):
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        device = "mps" if torch.backends.mps.is_available() else "cpu"

        print(f"Loading model on {device}...")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
        ).to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.device = device

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        text_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
            max_length=2500,
            do_sample=True,
            top_k=5,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        parser = JsonSchemaParser(schema.model_json_schema())
        prefix_function = build_transformers_prefix_allowed_tokens_fn(
            self.tokenizer, parser
        )

        output_dict = text_pipeline(prompt, prefix_allowed_tokens_fn=prefix_function)
        output = output_dict[0]["generated_text"][len(prompt):]
        json_result = json.loads(output)

        return schema(**json_result)

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "TinyLlama-1.1B-Chat-v1.0"
