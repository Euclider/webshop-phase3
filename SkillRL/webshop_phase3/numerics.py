"""Explicit new-cohort padding contract, applied to actor/reference/readout/SFT.

Qwen3.5's upstream batch-one fast path skips linear-attention padding masking.
This run-local patch makes a PAD prefix irrelevant; no site-packages are edited.
"""

VERSION = 'webshop.phase3.qwen35-mask-and-trim.v1'


def mask_padding(hidden_states, attention_mask):
    if attention_mask is not None and attention_mask.shape[1] > 1:
        if hidden_states.shape[:2] != attention_mask.shape:
            raise ValueError('Hidden/mask shape mismatch')
        return (hidden_states * attention_mask[:, :, None]).to(hidden_states.dtype)
    return hidden_states


def install():
    from transformers.models.qwen3_5 import modeling_qwen3_5
    modeling_qwen3_5.apply_mask_to_padding_states = mask_padding


def trim_inputs(inputs, response_length):
    from verl.workers.actor.padded_forward import trim_common_left_padding
    ids, mask, pos = trim_common_left_padding(inputs['input_ids'], inputs['attention_mask'],
                                             inputs['position_ids'], response_length)
    return {'input_ids': ids, 'attention_mask': mask, 'position_ids': pos}
