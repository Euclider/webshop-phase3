"""Three-arm orchestration contract with synthetic backends; NOT a GPU run."""
import json
from pathlib import Path
from test_bank_protocol import module


def test_three_arm_full_schedule_resume_does_not_repeat_updates_or_editor(tmp_path, monkeypatch):
    run=module('run');bank=module('bank').Bank.initial()
    from phase3.common import write_new,digest
    counts={'train':[],'edit':0,'score':[]}
    manifest={'root':str(tmp_path),'sft_model':'/shared/cold-start',
              'schedule':{'eval_ids':[0],'dev_ids':[500],'decoding_seeds':[40401]}}
    write_new(tmp_path/'preflight/complete.json',{'passed':True,'manifest_sha256':digest(manifest)})
    monkeypatch.setattr(module('preflight'),'verify_sources',lambda m:None)
    monkeypatch.setattr(module('retention'),'collect',lambda *a:None)
    monkeypatch.setattr(module('report'),'report',lambda *a:None)

    class Editor:
        def __init__(self,*a,**k):pass
        def __call__(self,payload,validate):
            counts['edit']+=1
            proposal={'operations':[{'op':'NOOP','targets':[],'skill':None,
                         'rationale':'Synthetic test only','evidence_ids':[]}]}
            validate(proposal)
            return proposal,{'api_calls':1}
        def close(self):pass
    monkeypatch.setattr(module('editor'),'Editor',Editor)

    def train(m,arm,start,current):
        counts['train'].append((arm,start,current.manifest_sha256))
        root=tmp_path/'runs'/arm;sid=current.skill_ids[0]
        for i in range(128):
            write_new(root/'episodes'/f'u{start+1:04d}'/'train'/f'e{i}.json',
                {'trajectory_id':f'e{i}','global_update':start+1,'sampling_policy_update':start,
                 'split':'train','task_id':1500+i//8,'task':'Synthetic purchase',
                 'bank_sha256':current.manifest_sha256,'success':False,'task_score':0.,
                 'steps':[{'selected_skill_id':sid,'skill_version_sha256':current.get(sid).payload_sha256}]})
        return {'model':str(root/'models'/f'u{start+5:04d}'),'native':'unused','update':start+5}
    monkeypatch.setattr(run,'train_window',train)

    def predict(m,root,start,old,new,current,output):
        counts['score'].append((start,old,new))
        assert old==('/shared/cold-start' if start==0 else str(root/'models'/f'u{start:04d}'))
        return {'start':start,'end':start+5,'bank_sha256':current.manifest_sha256,
                'rows':[{'skill_id':current.skill_ids[0],'n_loss_tokens':1,'D_sign_balance':1.}]}
    monkeypatch.setattr(run,'predict',predict)
    def evaluate(m,model,current,output,tasks):
        return [{'task_id':t,'eval_seed':40401,'task_score':0.,'success':False} for t in tasks]
    monkeypatch.setattr(run,'evaluation',evaluate)
    run.execute(manifest,module('protocol').ARMS)
    assert len(counts['train'])==90 and counts['edit']==60 and len(counts['score'])==30
    assert all(sha==bank.manifest_sha256 for _,_,sha in counts['train'])
    for arm in module('protocol').ARMS:
        assert len(list((tmp_path/'runs'/arm/'windows').glob('*/complete.json')))==30
        assert (tmp_path/'runs'/arm/'windows/u0145-u0150/complete.json').exists()
    run.execute(manifest,module('protocol').ARMS)
    assert len(counts['train'])==90 and counts['edit']==60
