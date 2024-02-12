# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import time
from threading import Thread
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers.generation.streamers import TextIteratorStreamer
from cog import BasePredictor, Input, ConcatenateIterator

# Enable faster download speed
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
CACHE_DIR = "model_cache"


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        model_name = "deepseek-ai/deepseek-math-7b-instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=CACHE_DIR,
        )
        self.model.generation_config = GenerationConfig.from_pretrained(
            model_name, cache_dir=CACHE_DIR
        )
        self.model.generation_config.pad_token_id = (
            self.model.generation_config.eos_token_id
        )

    def predict(
        self,
        text: str = Input(
            description="Input text.",
            default="what is the integral of x^2 from 0 to 2?\nPlease reason step by step, and put your final answer within \boxed{}.",
        ),
        max_new_tokens: int = Input(
            description="The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.",
            default=100,
        ),
        temperature: float = Input(
            description="The value used to modulate the next token probabilities.",
            default=1,
        ),
        top_k: int = Input(
            description="The number of highest probability vocabulary tokens to keep for top-k-filtering.",
            default=50,
        ),
        top_p: float = Input(
            description="If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.",
            default=0.9,
        ),
    ) -> ConcatenateIterator[str]:
        """Run a single prediction on the model"""

        messages = [{"role": "user", "content": text}]
        input_tensor = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        with torch.inference_mode():
            thread = Thread(
                target=self.model.generate,
                kwargs=dict(
                    input_ids=input_tensor.to(self.model.device),
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_new_tokens=max_new_tokens,
                    streamer=streamer,
                    use_cache=True,
                ),
            )
            thread.start()
            for new_token in streamer:
                yield new_token
            thread.join()
