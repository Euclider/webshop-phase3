import json
import pytest
from test_bank_protocol import module


def test_worker_environment_does_not_receive_editor_or_github_secrets(tmp_path,monkeypatch):
    monkeypatch.setenv('SKILLRL_PHASE3_EDITOR_API_KEY','test-secret')
    monkeypatch.setenv('GH_TOKEN','test-github')
    env=module('run').environment({'root':str(tmp_path),'base_model':'/raw','env_root':'/shop','router_gpus':[6,7]},
                                 '/bank','a'*64,tmp_path/'router.sqlite3')
    assert 'SKILLRL_PHASE3_EDITOR_API_KEY' not in env and 'GH_TOKEN' not in env
    assert env['WEBSHOP_BASE_MODEL']=='/raw' and env['WEBSHOP_PHASE3_BANK']=='/bank'


def test_grown_bank_router_exposes_new_candidate_and_isolates_cache(tmp_path):
    from webshop_phase12.llm_router import FrozenLLMSkillRouter
    from webshop_phase12.visible_state import VisibleMemory
    bank=module('bank').Bank.initial()
    patch={'operations':[{'op':'ADD','targets':[],'skill':{'name':'compare','description':'After visits','body':'Compare visible prices.'},
                          'rationale':'evidence','evidence_ids':['e']}]}
    grown,_=bank.apply(patch,event_id='u5',evidence_ids={'e'})
    class Backend:
        identity={'model':'synthetic-test-only'}
        def select_many(self,prompts):
            assert all(grown.skill_ids[-1] in p for p in prompts)
            return [{'label':'BC','input_tokens':100,'output_tokens':1,'latency_ms':0} for _ in prompts]
    router=FrozenLLMSkillRouter(tmp_path/'new.sqlite3',Backend(),grown,module('bank').labels_for_count(55))
    result=router.route_many([VisibleMemory('Buy shirt',50).state()])[0]
    assert result['selected_skill_id']==grown.skill_ids[-1]
    assert len(result['candidate_skill_ids'])==55
    assert router.protocol['bank_manifest_sha256']==grown.manifest_sha256


def test_sft_real_tokenizer_keeps_single_reasoning_opening():
    from transformers import AutoTokenizer
    from webshop_phase12.assets import BASE_MODEL
    tokenizer=AutoTokenizer.from_pretrained(BASE_MODEL,local_files_only=True)
    row=module('sft').encode_example({'instruction':'Buy a shirt','output':'<think>Check.</think>\n<action>click[x]</action>'},tokenizer)
    supervised=tokenizer.decode([x for x in row['labels'] if x!=-100])
    assert supervised.count('<think>')==1 and '<action>click[x]</action>' in supervised


def test_rollout_sampling_seed_depends_on_update_step_original_row_not_gpu_or_batch():
    seed=module('policy').rollout_seed
    assert seed(404,1,0,0)==404010000
    assert seed(404,5,49,127)==404056399
    assert len({seed(404,u,s,r) for u in (1,5,50) for s in range(50) for r in range(128)})==19200
