import json
import pytest
import torch
from test_bank_protocol import module


def test_checkpoint_retention_requires_sealed_window_and_preserves_latest(tmp_path):
    gc=module('retention').collect
    for p in ('models/u0005','models/u0010','checkpoints/global_step_9','checkpoints/global_step_10'):
        path=tmp_path/p;path.mkdir(parents=True);(path/'weights').write_text('evidence')
    with pytest.raises(ValueError):gc(tmp_path,10)
    seal=tmp_path/'windows/u0005-u0010';seal.mkdir(parents=True)
    (seal/'complete.json').write_text(json.dumps({'update':10,'model':str(tmp_path/'models/u0010')}))
    gc(tmp_path,10)
    assert not (tmp_path/'models/u0005').exists()
    assert (tmp_path/'models/u0010/weights').read_text()=='evidence'
    assert (tmp_path/'checkpoints/global_step_10/weights').exists()
    assert not (tmp_path/'checkpoints/global_step_9').exists()


def test_single_row_qwen35_padding_matches_unpadded_tiny_model():
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
    numerics=module('numerics');numerics.install()
    config=Qwen3_5TextConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
        num_attention_heads=2,num_key_value_heads=1,head_dim=16,linear_num_key_heads=2,
        linear_num_value_heads=2,linear_key_head_dim=16,linear_value_head_dim=16,
        layer_types=['linear_attention','full_attention'],attn_implementation='eager',pad_token_id=0)
    torch.manual_seed(404);model=Qwen3_5ForCausalLM(config).eval()
    ids=torch.tensor([[0,0,0,3,4,5,6]])
    mask=torch.tensor([[0,0,0,1,1,1,1]])
    pos=(mask.cumsum(-1)-1).clamp_min(0)
    with torch.no_grad():
        padded=model(input_ids=ids,attention_mask=mask,position_ids=pos,use_cache=False).logits[:,-4:]
        trimmed=model(input_ids=ids[:,3:],attention_mask=mask[:,3:],position_ids=pos[:,3:],use_cache=False).logits
    assert torch.allclose(padded,trimmed,atol=1e-5,rtol=1e-5)
