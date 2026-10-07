"""Portable source-only bundle with hashes; never include models or run artifacts."""
import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

PIN='f2dd4a14a15d1751c81da3fa64e4c4c7cbb205e9'
MANIFEST='WEBSHOP_PHASE3_BUNDLE.json'
PREFIX='webshop-phase3-b200'


def files(root):
    candidates=[root/name for name in ('README.md','THIRD_PARTY_NOTICES.md','.gitignore',
                'WEBSHOP_PHASE3_B200_START.md','WEBSHOP_PHASE3_START.md')]
    for folder in ('SkillRL','deploy/webshop'):
        candidates.extend((root/folder).rglob('*'))
    result=[]
    for path in sorted(set(candidates)):
        relative=path.relative_to(root)
        if any(p in ('.git','__pycache__','.pytest_cache','artifacts') or p.endswith('.egg-info') for p in relative.parts):continue
        if path.suffix in ('.pt','.pth','.safetensors','.ckpt','.bin','.pyc'):continue
        if path.is_symlink():raise ValueError('Bundle refuses symbolic links: '+str(relative))
        if path.is_file():result.append(path)
    return result


def build(root,output):
    root=Path(root).resolve();output=Path(output)
    entries=files(root)
    manifest={'schema':'webshop.phase3.source_bundle.v1','upstream_commit':PIN,
        'contains_local_phase3_changes':True,'training_started':False,
        'models_and_product_data_included':False,
        'files':[{'path':str(p.relative_to(root)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                  'bytes':p.stat().st_size} for p in entries]}
    with tarfile.open(output,'x:gz') as archive:
        for path in entries:archive.add(path,arcname=str(Path(PREFIX)/path.relative_to(root)),recursive=False)
        data=(json.dumps(manifest,indent=2)+'\n').encode()
        info=tarfile.TarInfo(PREFIX+'/'+MANIFEST);info.size=len(data);info.mode=0o644
        archive.addfile(info,io.BytesIO(data))
    return {'bundle':str(output),'files':len(entries),'bytes':output.stat().st_size,
            'sha256':hashlib.sha256(output.read_bytes()).hexdigest()}


def verify(root):
    root=Path(root).resolve();manifest=json.loads((root/MANIFEST).read_text())
    for row in manifest['files']:
        relative=Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe manifest path')
        path=root/relative
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('Bundle content mismatch: '+str(relative))
    return {'verified_files':len(manifest['files']),'upstream_commit':manifest['upstream_commit'],
            'gpu_training_validated':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['build','verify']);p.add_argument('--output')
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);a=p.parse_args()
    if a.mode=='build' and not a.output:p.error('build requires --output (new file)')
    print(json.dumps(build(a.root,a.output) if a.mode=='build' else verify(a.root),indent=2))
