"""Scoped, expiring experience rules stored locally in Chroma; no model updates."""
from time import time
from uuid import uuid4

import chromadb
from sentence_transformers import SentenceTransformer

_DEFAULT = object()

class ExperienceMemory:
    def __init__(self, path=None, model_name='sentence-transformers/all-MiniLM-L6-v2', revision=None, clock=None, encoder=None):
        self.encoder = encoder or SentenceTransformer(model_name, revision=revision, device='cpu')
        self.clock = clock or time
        self.client = chromadb.PersistentClient(path=str(path)) if path else chromadb.Client()
        self.collection = self.client.get_or_create_collection('cgr_experience', metadata={'hnsw:space': 'cosine'})

    def add_rule(self, task_description=None, mistake=None, rule=None, *, rule_text=None, scope, tier, ttl_seconds=_DEFAULT):
        # The older (task, mistake, rule) form remains usable with explicit scope and tier.
        if rule_text is not None:
            if task_description is not None or rule is not None: raise ValueError('use either rule_text or the legacy arguments')
            task_description = rule_text
            rule = rule_text
        elif rule is None and mistake is None:
            rule = task_description
        if not all(isinstance(x, str) and x.strip() for x in (task_description, rule, scope)):
            raise ValueError('task description, rule and scope must be nonempty strings')
        if not isinstance(tier, int) or isinstance(tier, bool) or tier not in (1,2,3): raise ValueError('tier must be 1, 2 or 3')
        if tier == 1 and scope != 'global': raise ValueError('tier 1 must have global scope')
        if tier == 2 and scope == 'global': raise ValueError('tier 2 requires a non-global scope')
        if ttl_seconds is _DEFAULT: ttl_seconds = {1: None, 2: 3600, 3: 40*86400}[tier]
        if tier == 1 and ttl_seconds is not None: raise ValueError('tier 1 is permanent; delete it manually')
        if tier in (2,3) and ttl_seconds is None: raise ValueError('tier 2 and 3 require expiry')
        if ttl_seconds is not None and (isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, (int,float)) or not 0 < ttl_seconds < float('inf')):
            raise ValueError('ttl_seconds must be positive and finite')
        now = float(self.clock())
        expires = 0.0 if ttl_seconds is None else now + ttl_seconds
        self.collect_expired(now)
        vector = self.encoder.encode(task_description, normalize_embeddings=True).tolist()
        rule_id = str(uuid4())
        self.collection.add(ids=[rule_id], embeddings=[vector], documents=[task_description], metadatas=[{'mistake': mistake or '', 'rule': rule, 'scope': scope, 'tier': tier, 'created_at': now, 'expires_at': expires}])
        return rule_id

    def collect_expired(self, now=None):
        now = float(self.clock()) if now is None else now
        records = self.collection.get(include=['metadatas'])
        # Old records without explicit metadata are excluded rather than promoted to global.
        ids = [i for i,m in zip(records['ids'], records['metadatas']) if m and m.get('expires_at',0) > 0 and m['expires_at'] <= now]
        if ids: self.collection.delete(ids=ids)
        return len(ids)

    def delete_rule(self, rule_id):
        self.collection.delete(ids=[rule_id])

    def end_session(self, scope):
        if not scope or scope == 'global': raise ValueError('a non-global session scope is required')
        records = self.collection.get(where={'$and':[{'scope':scope},{'tier':2}]}, include=[])
        if records['ids']: self.collection.delete(ids=records['ids'])
        return len(records['ids'])

    def retrieve(self, task_description, top_k=2, *, scope=None):
        if not isinstance(task_description,str) or not task_description.strip() or not isinstance(top_k,int) or isinstance(top_k,bool) or top_k < 1: raise ValueError('provide a nonempty task and a positive top_k')
        if scope is not None and (not isinstance(scope,str) or not scope.strip()): raise ValueError('scope must be a nonempty string')
        self.collect_expired()
        where = {'scope':'global'} if scope is None or scope == 'global' else {'$or':[{'scope':'global'},{'scope':scope}]}
        eligible = self.collection.get(where=where, include=['metadatas'])
        n = min(top_k, len(eligible['ids']))
        if not n: return []
        vector = self.encoder.encode(task_description, normalize_embeddings=True).tolist()
        result = self.collection.query(query_embeddings=[vector], n_results=n, where=where, include=['documents','metadatas','distances'])
        return [{'task':d, 'mistake':m['mistake'], 'rule':m['rule'], 'scope':m['scope'], 'tier':m['tier'], 'created_at':m['created_at'], 'expires_at':None if m['expires_at']==0 else m['expires_at'], 'distance':float(s)} for d,m,s in zip(result['documents'][0],result['metadatas'][0],result['distances'][0])]

    def get_rules(self, task_description, top_k=2, *, scope=None):
        rules = self.retrieve(task_description, top_k, scope=scope)
        return '[SYSTEM MEMORY: ' + ' | '.join(r['rule'] for r in rules) + ']' if rules else ''