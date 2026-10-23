from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

def build_bm_25_retriever(documents:list, k:int=5)-> BM25Retriever:
    bm25_retriever = BM25Retriever.from_documents(documents=documents,
                                                search_kwargs = {"k":k})
    return bm25_retriever

def build_ensemble_retriever(bm25_retriever, vector_retriever, weights):
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=weights)
    return ensemble_retriever


