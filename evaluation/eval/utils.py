import torch
import tqdm
from transformers import StoppingCriteria, GenerationConfig

class KeyWordsCriteria(StoppingCriteria):
    def __init__(self, stop_id_sequences, tokenizer, prompt_length):
        assert isinstance(stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
        self.tokenizer = tokenizer
        self.stop_id_sequences = stop_id_sequences
        self.stop_sequences = [tokenizer.decode(sequence) for sequence in stop_id_sequences]
        print(f"stop sequences: {self.stop_sequences}", flush=True)
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            ids = input_ids[i][self.prompt_length:].tolist()
            should_be_stopped = False
            for stop_ids, stop_sequence in zip(self.stop_id_sequences, self.stop_sequences):
                _ids = ids
                for j in range(len(_ids), 0, -1):
                    s = self.tokenizer.decode(_ids[max(j - len(stop_ids) - 3, 0) :j])
                    if s.endswith(stop_sequence):
                        should_be_stopped = True
                        break
                if should_be_stopped:
                    break
            sequences_should_be_stopped.append(should_be_stopped)
        return all(sequences_should_be_stopped)
    
@torch.no_grad()
def generate_completions(model, tokenizer, prompts, batch_size=1, stop_id_sequences=None, end_of_generation_id_sequence=None, disable_tqdm=False, **generation_kwargs):
    generations = []
    finish_completion = []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Generating Completions")

    if stop_id_sequences is not None:
        stop_sequences = [tokenizer.decode(stop_id_sequence) for stop_id_sequence in stop_id_sequences]

    if end_of_generation_id_sequence is not None:
        end_of_generation_sequence = tokenizer.decode(end_of_generation_id_sequence)

    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
    generation_kwargs['use_cache'] = True
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens='chatglm2' in str(model.__class__))
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        if model.device.type == "cuda":
            batch_input_ids = batch_input_ids.cuda()
            attention_mask = attention_mask.cuda()

        batch_finish_completion = [False] * len(batch_prompts) * num_return_sequences
        try:
            batch_outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=attention_mask,
                stopping_criteria=[KeyWordsCriteria(stop_id_sequences, tokenizer, batch_input_ids.size(1))] if stop_id_sequences else None,
                **generation_kwargs
            )

            # the stopping criteria is applied at batch level, so if other examples are not stopped, the entire batch will continue to generate.
            # so some outputs still have the stop sequence, which we need to remove.
            if stop_id_sequences:
                for output_idx in range(batch_outputs.shape[0]):
                    finish = False
                    for token_idx in range(batch_input_ids.shape[1], batch_outputs.shape[1]):
                        if any(tokenizer.decode(batch_outputs[output_idx, token_idx: token_idx + len(stop_sequence) + 3]).startswith(stop_sequence) for stop_sequence in stop_sequences):
                            if end_of_generation_id_sequence is not None and tokenizer.decode(batch_outputs[output_idx, token_idx: token_idx + len(end_of_generation_id_sequence) + 3]).startswith(end_of_generation_sequence):
                                batch_finish_completion[output_idx] = True
                            batch_outputs[output_idx, token_idx:] = tokenizer.pad_token_id
                            break

            # remove the prompt from the output
            # we need to re-encode the prompt because we need to make sure the special tokens are treated the same way as in the outputs.
            # we changed our previous way of truncating the output token ids dicrectly because some tokenizer (e.g., llama) won't add space token before the first token.
            # space is important for some tasks (e.g., code completion).
            batch_outputs = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
            batch_prompts = tokenizer.batch_decode(batch_input_ids, skip_special_tokens=True)
            # duplicate the prompts to match the number of return sequences
            batch_prompts = [prompt for prompt in batch_prompts for _ in range(num_return_sequences)]
            batch_generations = [
                output[len(prompt):] for prompt, output in zip(batch_prompts, batch_outputs)
            ]
        except Exception as e:
            print("Error when generating completions for batch:")
            print(batch_prompts)
            print("Error message:")
            print(e)
            print("Use empty string as the completion.")
            batch_generations = [""] * len(batch_prompts) * num_return_sequences

        generations += batch_generations
        finish_completion += batch_finish_completion

        if not disable_tqdm:
            progress.update(len(batch_prompts)//num_return_sequences)

    assert len(generations) == len(prompts) * num_return_sequences, "number of generations should be equal to number of prompts * num_return_sequences"
    return generations, finish_completion


@torch.no_grad()
def get_next_word_predictions(model, tokenizer, prompts, candidate_token_ids=None, batch_size=1, return_token_predictions=False, disable_tqdm=False):
    predictions, probs = [], []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Getting Predictions")

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i: i+batch_size]
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=False)
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        if model.device.type == "cuda":
            batch_input_ids = batch_input_ids.cuda()
            attention_mask = attention_mask.cuda()

        batch_logits = model(input_ids=batch_input_ids, attention_mask=attention_mask).logits[:, -1, :]
        if candidate_token_ids is not None:
            batch_logits = batch_logits[:, candidate_token_ids]
        batch_probs = torch.softmax(batch_logits, dim=-1)
        batch_prediction_indices = torch.argmax(batch_probs, dim=-1)
        if return_token_predictions:
            if candidate_token_ids is not None:
                candidate_tokens = tokenizer.convert_ids_to_tokens(candidate_token_ids)
                batch_predictions = [candidate_tokens[idx] for idx in batch_prediction_indices]
            else:
                batch_predictions = tokenizer.convert_ids_to_tokens(batch_prediction_indices)
            predictions += batch_predictions
        else:
            predictions += batch_prediction_indices.tolist()
        probs += batch_probs.tolist()

        if not disable_tqdm:
            progress.update(len(batch_prompts))

    assert len(predictions) == len(prompts), "number of predictions should be equal to number of prompts"
    return predictions, probs


@torch.no_grad()
def score_completions(model, tokenizer, scoring_examples, disable_tqdm=False):
    '''
    Each scoring example is a dict, which contains the following keys:
    - prompt: the prompt to score
    - completions: a list of completions to score
    '''
    
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(scoring_examples), desc="Scoring Completions")

    # unroll the scoring examples
    unrolled_examples = []
    for scoring_example in scoring_examples:
        prompt = scoring_example["prompt"]
        for completion in scoring_example["completions"]:
            unrolled_examples.append({
                "prompt": prompt,
                "completion": completion
            })

    scores = []
    # currently we don't support batching, because we want to directly use the loss returned by the model to score each completion.
    for unrolled_example in unrolled_examples:
        encoded_example = encode_with_prompt_completion_format(unrolled_example, tokenizer, max_seq_length=None)
        # unsqueeze the batch dimension
        for key, value in encoded_example.items():
            encoded_example[key] = value.unsqueeze(0)
        if model.device.type == "cuda":
            encoded_example = {
                key: value.cuda() for key, value in encoded_example.items()
            }
        outputs = model(**encoded_example)
        loss = outputs.loss
        scores.append(-loss.item())
        if not disable_tqdm:
            progress.update(1)

    # roll up the scores
    rolled_up_scores = {}
    for unrolled_example, score in zip(unrolled_examples, scores):
        prompt = unrolled_example["prompt"]
        completion = unrolled_example["completion"]
        if prompt not in rolled_up_scores:
            rolled_up_scores[prompt] = {}
        rolled_up_scores[prompt][completion] = score

    return rolled_up_scores



def load_hf_lm_and_tokenizer(
        model_name_or_path, 
        tokenizer_name_or_path=None, 
        device_map="auto", 
        load_in_8bit=False, 
        load_in_half=False,
        gptq_model=False,
        use_fast_tokenizer=True,
        padding_side="left",
    ):
    
    from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer

    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_name_or_path

    is_chatglm2 = 'chatglm2' in tokenizer_name_or_path.lower() or 'chatglm2' in model_name_or_path
    is_qwen = 'qwen' in tokenizer_name_or_path.lower() or 'qwen' in model_name_or_path

    if is_chatglm2 or is_qwen:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
        if is_qwen:
            tokenizer.eos_token = '<|endoftext|>'
            tokenizer.eos_token_id = 151643
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True, use_fast=use_fast_tokenizer)
    # set padding side to left for batch generation
    tokenizer.padding_side = padding_side
    # set pad token to eos token if pad token is not set (as is the case for llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if gptq_model:
        from auto_gptq import AutoGPTQForCausalLM
        model_wrapper = AutoGPTQForCausalLM.from_quantized(
            model_name_or_path, device="cuda:0", use_triton=True
        )
        model = model_wrapper.model  
    elif load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            device_map=device_map, 
            load_in_8bit=True
        )
    else:
        kwargs = {}
        model_class = AutoModelForCausalLM
        if is_chatglm2:
            kwargs = {'trust_remote_code': True}
            model_class = AutoModel
        elif is_qwen:
            kwargs = {'trust_remote_code': True}
        if device_map:
            model = model_class.from_pretrained(model_name_or_path, device_map=device_map, **kwargs)
        else:
            model = model_class.from_pretrained(model_name_or_path, **kwargs)
            if torch.cuda.is_available():
                model = model.cuda()
        if is_qwen:
            model.generation_config = GenerationConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
            model.generation_config.do_sample = False
        if not is_chatglm2 and not is_qwen and load_in_half:
            model = model.half()
    model.eval()
    return model, tokenizer
