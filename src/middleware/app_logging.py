from langchain_core.callbacks import BaseCallbackHandler
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

class Logger(BaseCallbackHandler):
    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id = None, tags = None, metadata = None, **kwargs):
        return super().on_chain_start(serialized, inputs, run_id=run_id, parent_run_id=parent_run_id, tags=tags, metadata=metadata, **kwargs)
    
    def on_chain_end(self, outputs, *, run_id, parent_run_id = None, **kwargs):
        return super().on_chain_end(outputs, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
    
    def on_retriever_end(self, documents, *, run_id, parent_run_id = None, **kwargs):
        return super().on_retriever_end(documents, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
    
    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id = None, tags = None, metadata = None, **kwargs):
        return super().on_llm_start(serialized, prompts, run_id=run_id, parent_run_id=parent_run_id, tags=tags, metadata=metadata, **kwargs)
    
    def on_llm_error(self, error, *, run_id, parent_run_id = None, tags = None, **kwargs):
        return super().on_llm_error(error, run_id=run_id, parent_run_id=parent_run_id, tags=tags, **kwargs)
    
@dataclass
class RequestMetrics:
    session_id:str
    questions:str
    nb_chunks_retrieved:int 
    latency_ms:float 
    cache_hit:bool 
    model_used:str 
    timestamp:str=None
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()
def log_metrics(metrics:RequestMetrics, filepath:str)-> None:
    path= Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics_dict = asdict(metrics)
    with open(path, mode='a', encoding='utf-8') as f:
        f.write(json.dumps(metrics_dict, ensure_ascii=False) + "\n")
        
    