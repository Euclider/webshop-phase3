"""Release only this run's reproducible model/checkpoint copies after sealing."""
import json
import re
import shutil
from pathlib import Path
from phase3.common import require, write_new
from .protocol import UPDATES, WINDOW


def collect(root, update):
    root=Path(root).resolve()
    require(update in range(WINDOW,UPDATES+1,WINDOW),'Invalid sealed endpoint')
    seal=root/'windows'/f'u{update-5:04d}-u{update:04d}'/'complete.json'
    require(seal.is_file(),'Cannot release unsealed window dependencies')
    record=json.loads(seal.read_text())
    require(record['update']==update and Path(record['model']).resolve()==root/'models'/f'u{update:04d}',
            'Foreign sealed model')
    audit=root/'retention'/f'u{update:04d}.json'
    if audit.exists():
        plan=json.loads(audit.read_text())
    else:
        paths=[]
        for folder,pattern in (('models',r'u(\d{4})'),('checkpoints',r'global_step_(\d+)')):
            for path in sorted((root/folder).glob('*')):
                match=re.fullmatch(pattern,path.name)
                if match and int(match[1])<update:
                    require(path.is_dir() and not path.is_symlink(),'Unsafe checkpoint target')
                    paths.append(str(path.relative_to(root)))
        plan={'update':update,'root':str(root),'released_paths':paths,
              'reason':'window readout, gate and compact evidence sealed; retain latest native and HF model'}
        write_new(audit,plan)
    require(plan['root']==str(root) and plan['update']==update,'Foreign retention plan')
    for relative in plan['released_paths']:
        path=root/relative
        require(path.parent in (root/'models',root/'checkpoints') and not path.is_symlink()
                and path.resolve().is_relative_to(root),'Unsafe recovery retention path')
        if path.exists():shutil.rmtree(path)
    return plan
