"""vLLM policy evaluation with explicit per-episode decoding seeds."""
from phase3.common import require
from .protocol import UPDATES


def rollout_seed(seed, update, step, original_row):
    require(seed==404 and 1<=update<=UPDATES and 0<=step<50 and 0<=original_row<128, 'Invalid sampling identity')
    return seed*1000000+update*10000+step*128+original_row


def prepare_request(tokenizer, request):
    rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': request['prompt']}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    require(0 < len(ids) <= 16384 and 0 < request['max_new_tokens'] <= 512, 'Policy context overflow; no truncation')
    return {'prompt_token_ids': ids}, dict(n=1, seed=int(request['seed']),
        temperature=float(request['temperature']), top_p=1., max_tokens=int(request['max_new_tokens']), detokenize=True)


class Policy:
    def __init__(self, model):
        from skillnet_cohort.inference import WEBSHOP_B200_PROFILE, registration
        from skillnet_cohort.vllm_backend import build_engine
        self.engine = build_engine(model, registration(WEBSHOP_B200_PROFILE))
        self.tokenizer = self.engine.get_tokenizer()

    def generate_batch(self, requests):
        from vllm import SamplingParams
        values = [prepare_request(self.tokenizer, r) for r in requests]
        out = self.engine.generate([p for p, _ in values], [SamplingParams(**s) for _, s in values], use_tqdm=False)
        require(len(out) == len(requests), 'Dropped generation requests')
        return [(x.outputs[0].text, len(x.prompt_token_ids), len(x.outputs[0].token_ids)) for x in out]

    def close(self): self.engine.llm_engine.engine_core.shutdown(timeout=20)
