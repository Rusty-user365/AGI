"""Exercise scope isolation and TTL using a fake clock and the real embedding model."""
from pathlib import Path
from contextlib import nullcontext
from experience_memory import ExperienceMemory

class Clock:
    now = 10000.0
    def __call__(self): return self.now

def main():
    clock=Clock()
    model=Path(__file__).resolve().parent/'.experience_memory_models/all-MiniLM-L6-v2'
    with nullcontext(None) as directory:
        memory=ExperienceMemory(path=directory,model_name=str(model),clock=clock)
        memory.add_rule('Format as Answer: A',scope='cyber_test',tier=2,ttl_seconds=60)
        immediate=memory.get_rules('Cyber test question 2',scope='cyber_test')
        assert 'Answer: A' in immediate
        poem=memory.get_rules('Write a poem')
        assert 'Answer: A' not in poem
        other_scope=memory.get_rules('Cyber test question 2',scope='other_session')
        assert 'Answer: A' not in other_scope
        print('Immediate:', immediate)
        print('Poem:', poem or '(no memory injected)')
        print('Other scope:', other_scope or '(no memory injected)')
        clock.now+=60
        expired=memory.get_rules('Cyber test question 3',scope='cyber_test')
        assert 'Answer: A' not in expired and memory.collection.count()==0
        print('At expiry:', expired or '(no memory injected)')
        memory.add_rule('Global rule',scope='global',tier=1,ttl_seconds=None)
        memory.add_rule('Time-sensitive documentation',scope='docs',tier=3)
        clock.now+=40*86400
        assert 'Global rule' in memory.get_rules('Global rule')
        assert 'Time-sensitive documentation' not in memory.get_rules('documentation',scope='docs')
        memory.add_rule('Session reminder',scope='session_123',tier=2)
        assert memory.end_session('session_123')==1
        assert 'Session reminder' not in memory.get_rules('Session reminder',scope='session_123')
        try: memory.add_rule('bad scope',scope='global',tier=2,ttl_seconds=10)
        except ValueError: pass
        else: raise AssertionError('tier 2 cannot be global')
        print('PASS: expiry boundary, global permanence, 40-day expiry, session close and scope validation')

if __name__=='__main__': main()