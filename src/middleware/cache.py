from langchain_community.cache import InMemoryCache, RedisSemanticCache
import langchain
import uuid
import chromadb
from sentence_transformers import SentenceTransformer 

def setup_exact_cache():
    """Configure un cache exact en mémoire. Deux requêtes doivent être mot à mot identiques pour déclencher un 'Cache Hit' """
    langchain.llm_cache = InMemoryCache()
    
def setup_semantic_cache(redis_url, embeddings, threshold:float= 0.95)-> None:
    """Configure un cache sémantique basé sur Redis; Les requêtes ayant un sens similaire (dépassant le seuil 'threshold' déclencheront un 'Cache Hit')"""
    langchain.llm_cache = RedisSemanticCache(
        redis_url=redis_url,
        embedding=embeddings,
        score_threshold=threshold
    )

class ChromaSemanticCache:
    def __init__(self, collection_name = "llm_semantic_cache", similarity_threshold = 0.85, persist_directory= "./chroma_cache" ):
        self.client = chromadb.PersistentClient(path = persist_directory)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.collection = self.client.get_or_create_collection(
            name = collection_name,
            metadata = {"hnsw:space":"cosine"}
        )
        self.threshold = similarity_threshold
    def get(self,prompt:str):
        query_embedding = self.encoder.encode(prompt).tolist()
        results = self.collection.query(
            query_embeddings = [query_embedding], 
            n_results = 1,
            include = ["documents", "metadatas", "distances"]
        )
        if results["ids"] and len(results["ids"][0])>0:
            distance = results["distances"][0][0]
            similarity = 1.0 - distance
            
            if similarity>=self.threshold:
                cached_prompt = results["documents"][0][0]
                cached_response= results["metadatas"][0][0]["response"]
                return {
                    "hit":True,
                    "similarity":round(similarity, 4),
                    "original_prompt": cached_prompt,
                    "response":cached_response
                }
        return {"hit":False, "response":None}
    def set(self, prompt:str, response:str):
        prompt_embedding = self.encoder.encode(prompt).tolist()
        doc_id = str(uuid.uuid4())
        self.collection.add(
            ids = [doc_id],
            embeddings = [prompt_embedding], 
            documents = [prompt],
            metadatas=[{"response":response}]
        )
        
        