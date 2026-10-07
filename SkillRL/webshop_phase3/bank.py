"""Immutable WebShop banks; initial payload bytes/order remain unchanged."""
import hashlib
import itertools
import json
import string
from dataclasses import asdict
from pathlib import Path

from webshop_phase12.assets import Skill, WebshopBank, digest
from phase3.common import write_new, require


def labels_for_count(count):
    require(type(count) is int and count > 0, 'Empty label catalog')
    result = []
    for width in range(1, 5):
        for chars in itertools.product(string.ascii_uppercase, repeat=width):
            result.append(''.join(chars))
            if len(result) == count:
                return tuple(result)
    raise ValueError('Label capacity exhausted')


def validated_labels(tokenizer, count):
    """Preserve original 54 labels; expand with distinct single-token labels."""
    original = labels_for_count(54)
    selected, seen = [], set()
    for label in labels_for_count(18278):
        ids = tokenizer.encode(label, add_special_tokens=False)
        valid = len(ids) == 1 and ids[0] not in seen
        if label in original and not valid:
            raise ValueError('Original 54-label tokenizer contract failed')
        if valid:
            selected.append(label); seen.add(ids[0])
        if len(selected) == count:
            return tuple(selected)
    raise ValueError('Insufficient distinct single-token router labels')


class Bank:
    payload_renderer = 'skillscope.webshop.phase3.stored_payload.v1'

    def __init__(self, skills, *, parent=None, event='initial', retired=(), source=None):
        self.skills = tuple(skills)
        self.parent, self.event, self.retired, self.source = parent, event, tuple(retired), source
        require(self.skills and len(set(self.skill_ids)) == len(self.skills), 'Empty/duplicate bank')
        require(not set(self.skill_ids) & set(retired), 'Retired ID reused')
        for skill in self.skills:
            require(all(isinstance(v, str) and v.strip() for v in asdict(skill).values()), 'Invalid skill')
            require(hashlib.sha256(skill.payload.encode()).hexdigest() == skill.payload_sha256, 'Invalid payload hash')

    @classmethod
    def initial(cls):
        source = WebshopBank()
        return cls([source.get(s) for s in source.skill_ids], source=source.manifest_sha256)

    @property
    def skill_ids(self): return tuple(s.skill_id for s in self.skills)
    @property
    def manifest_sha256(self): return digest(self.record())
    @property
    def content_sha256(self): return digest([asdict(s) for s in self.skills])
    @property
    def bank_id(self): return 'webshop-phase3:' + self.manifest_sha256
    @property
    def manifest(self): return self.record()
    def __len__(self): return len(self.skills)
    def get(self, sid):
        for skill in self.skills:
            if skill.skill_id == sid: return skill
        raise KeyError(sid)
    def router_catalog(self):
        return [{'skill_id': s.skill_id, 'name': s.name, 'description': s.description} for s in self.skills]
    def record(self):
        return {'schema': 'webshop.phase3.bank.v1', 'skills': [asdict(s) for s in self.skills],
                'parent': self.parent, 'event': self.event, 'retired': list(self.retired), 'source': self.source}
    def save(self, root):
        return write_new(Path(root) / (self.manifest_sha256 + '.json'), self.record())
    @classmethod
    def load(cls, path, expected):
        record = json.loads(Path(path).read_text())
        require(digest(record) == expected and record['schema'] == 'webshop.phase3.bank.v1', 'Changed bank snapshot')
        bank = cls([Skill(**s) for s in record['skills']], parent=record['parent'], event=record['event'],
                   retired=record['retired'], source=record['source'])
        require(bank.record() == record, 'Noncanonical bank record')
        return bank

    def apply(self, patch, *, event_id, evidence_ids, allowed_ids=None):
        require(set(patch) == {'operations'}, 'Invalid patch schema')
        ops = patch['operations']
        require(isinstance(ops, list) and 1 <= len(ops) <= 3, 'One to three operations required')
        active = {s.skill_id: s for s in self.skills}
        allowed = set(active) if allowed_ids is None else set(allowed_ids)
        retired, touched = list(self.retired), set()
        counts = dict.fromkeys(('ADD', 'MODIFY', 'DELETE', 'MERGE', 'NOOP', 'mutation_units'), 0)
        for i, op in enumerate(ops):
            require(set(op) == {'op', 'targets', 'skill', 'rationale', 'evidence_ids'}, 'Invalid operation schema')
            kind, targets = op['op'], op['targets']
            require(kind in counts and kind != 'mutation_units' and isinstance(targets, list), 'Invalid operation')
            require(isinstance(op['rationale'], str) and op['rationale'].strip(), 'Missing rationale')
            require(isinstance(op['evidence_ids'], list) and set(op['evidence_ids']) <= set(evidence_ids), 'Foreign evidence')
            ids = [t['skill_id'] for t in targets]
            require(len(set(ids)) == len(ids) and not set(ids) & touched, 'Duplicate target')
            require(set(ids) <= allowed and set(ids) <= set(active), 'Noncandidate/inactive target')
            for target in targets:
                require(set(target) == {'skill_id', 'version_sha256'} and
                        target['version_sha256'] == active[target['skill_id']].payload_sha256, 'Stale target version')
            if kind == 'NOOP':
                require(len(ops) == 1 and not targets and op['skill'] is None, 'Invalid NOOP')
                counts['NOOP'] = 1
                return self, counts
            require(op['evidence_ids'], 'Mutation needs evidence')
            require((kind == 'ADD' and not ids) or (kind in ('MODIFY', 'DELETE') and len(ids) == 1)
                    or (kind == 'MERGE' and len(ids) >= 2), 'Wrong operation arity')
            units = len(ids) + 1 if kind == 'MERGE' else 1
            counts['mutation_units'] += units
            require(counts['mutation_units'] <= 3, 'Mutation budget exceeded')
            counts[kind] += 1; touched.update(ids)
            if kind in ('DELETE', 'MERGE'):
                for sid in ids: del active[sid]; retired.append(sid)
            if kind == 'DELETE':
                require(op['skill'] is None, 'DELETE cannot introduce content')
                continue
            content = op['skill']
            require(isinstance(content, dict) and set(content) == {'name', 'description', 'body'} and
                    all(isinstance(x, str) and x.strip() for x in content.values()), 'Invalid new skill')
            sid = ids[0] if kind == 'MODIFY' else 'ws-' + digest([self.manifest_sha256, event_id, i])[:20]
            payload = f"{content['name']}\nPrinciple: {content['body']}\nWhen to apply: {content['description']}"
            active[sid] = Skill(sid, content['name'], content['description'], payload, hashlib.sha256(payload.encode()).hexdigest())
        return Bank(active.values(), parent=self.manifest_sha256, event=event_id,
                    retired=retired, source=self.source), counts
