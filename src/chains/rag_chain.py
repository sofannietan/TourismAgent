from langchain_core.output_parsers import StrOutputParser
import asyncio
from src.retrieval.hybrid import *
from typing import AsyncIterator, Union
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from config import Settings
from src.retrieval.query_expansion import *
from src.middleware.app_logging import *
from src.middleware.guardrails import *
from src.middleware.cache import *
from src.agents.tools import *
from src.agents.agent import *
from src.memory.memory_manager import *
from langchain.chat_models import init_chat_model
from src.ingestion.embedder import *
from src.ingestion.chunker import *
from src.ingestion.loader import *
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import warnings
warnings.filterwarnings("ignore")
import asyncio
import sys
import uuid
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table
import os
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONWARNINGS"] = "ignore"

import warnings
import logging

warnings.filterwarnings("ignore")
for name in ("pypdf", "PyPDF2", "pdfminer", "transformers", "sentence_transformers", "httpx"):
    logging.getLogger(name).setLevel(logging.ERROR)


from langchain_core._api import LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
class TourismAgentPipeline:
    os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY 
    def __init__(self, session_id :str = "default_session"):
        self.session_id = session_id
        

        self.embeddings = build_embeddings(settings.GOOGLE_API_KEY, model_name = settings.embedding_model)
        self.documents = self._build_documents()
        self.chunks = chunking(self.documents, settings.chunk_size, settings.chunk_overlap)
        self.vectorstore = build_vectorstore(self.chunks, self.embeddings)
        self.logger = Logger()
        self.llm = ChatGoogleGenerativeAI(
                    model=settings.model_name,
                    callbacks=[self.logger])
        
        
        self.retriever = self._build_retriever()
        self.memory = build_local_memory(self.llm)
        self.tools = build_tools(self.retriever, self.vectorstore, self.memory)
        self.agent = build_agent(self.llm, self.tools, callbacks = [self.logger])
        self.rag_chain = self._build_rag_chain()
        # self.structured_chain = build_structured_rag_chain(self.llm, self.retriever) 
        self.router = build_query_router(self.llm)
        self.cache = ChromaSemanticCache(similarity_threshold = 0.85)
    def _build_retriever(self):
        """Construit le retreiver hybride BM25 + vectorstore avec Multiquery/qeury expansion"""
        all_docs = self.documents
        self.bm25_retriever = build_bm_25_retriever(all_docs, k = settings.top_k_retrieval)
        self.vector_retriever = self.vectorstore.as_retriever(search_kwargs = {"k":settings.top_k_retrieval})
        self.ensemble_retriever = build_ensemble_retriever(self.bm25_retriever, self.vector_retriever, weights = [0.5,0.5])
        # self.multiquery_retriever = build_multiquery_retriever(, self.llm)
        return self.ensemble_retriever
    
    def _build_documents(self):
        documents=[]
        for file_path in settings.file_paths:
            documents.append(load_documents(file_path))
        documents=[doc for sublist in documents for doc in sublist]
        return documents
    def _build_rag_chain(self):
        contextualize_q_system_prompt = (
            "Given the chat history and the user's last question "
            "that could refer to the history context, reformulate a question "
            "that could be understood without the chat history. Don't answer it. "
            "Just reformulate if necessary or return it as it was written"
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            ('placeholder', "{chat_history}"),
            ("human", "{question}"),
        ])
        contextualize_q_chain = contextualize_q_prompt | self.llm | StrOutputParser()
        
        # Fonction sécurisée pour récupérer le contexte
        def get_context(input_dict):
            # S'assure que input_dict est bien un dictionnaire
            if not isinstance(input_dict, dict):
                input_dict = {"question": str(input_dict), "chat_history": []}
                
            history = input_dict.get("chat_history", [])
            if history:
                reformulated = contextualize_q_chain.invoke(input_dict)
                docs = self.retriever.invoke(reformulated)
            else:
                docs = self.retriever.invoke(input_dict.get("question", ""))
            
            return "\n\n".join([f"[Source : {d.metadata.get('source', 'Inconnue')}]\n{d.page_content}" for d in docs])

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are TourismAgent, an expert of tourism in Cambodia. Answer the most precisely with the context and history. 
            Context : {context}"""),
            ("placeholder", "{chat_history}"),
            ("human", "{question}")
        ])
        
        # Construction propre sans RunnablePassthrough.assign pour éviter les pièges de types
        core_chain = (
            {
                "context": RunnableLambda(get_context),
                "question": lambda x: x.get("question") if isinstance(x, dict) else str(x),
                "chat_history": lambda x: x.get("chat_history", []) if isinstance(x, dict) else []
            }
            | qa_prompt
            | self.llm 
            | StrOutputParser()
        )

        chain_with_guardrails = (
            RunnableLambda(input_guardrail)
            | core_chain
        )
        return build_chain_with_history(chain_with_guardrails)
        
    def query(self, question: str, mode:str="auto")-> Union[str, dict]:
        """
        Mode 'auto' : uses the router to choose between RAG and agent
        Mode 'RAG' : executes the RAG chain
        Mode 'agent' : executes ReAct agent
        """
        cache_res = self.cache.get(question)
        if cache_res["hit"]:
            return cache_res["response"]
        if mode =='auto':
            mode = router_query(question,self.router)
        if mode =='agent':
            response = self.agent.invoke(
                {"messages":[("human", question)]}, 
                config = {"configurable":{"session_id":self.session_id}}
            )
            if isinstance(response, dict) and "messages" in response:
                final_response = response["messages"][-1].content
            else: 
                final_response = response
        else : 
            final_response = self.rag_chain.invoke(
                {"question":question},
                config = {"configurable":{"session_id":self.session_id}}
            )
        if isinstance(final_response,str):
            self.cache.set(question, final_response)
        return final_response
    
    def stream_events(self,question:str)-> None:
        try:
            for chunk in self.rag_chain.stream(
               {"question":question},
                config = {"configurable":{"session_id":self.session_id}}
                           ):
                print(chunk, end="", flush = True)
        except Exception as e:
            print(f"Error during streaming {e}")
    async def stream_agent_events(self, question:str)-> None:
        async for event in self.agent.astream_events(
            {"messages":[("user", question)]},
            config = {"configurable":{"session_id":self.session_id}}, 
            version = "v2"
                                       ):
            kind = event['event']
            
            if kind =="on_tool_start":
                tool_name = event["name"]
                tool_input =event["data"].get("input")
                print(f"Tool call : {tool_name}")
                print(f"Tool input : {tool_input}")
            elif kind =="on_tool_end":
                tool_output =event["data"].get("output")
                print(f"Tool output : {tool_output}")
            elif kind =="on_chat_model_stream":
                chunk =event["data"]["chunk"]
                if hasattr(chunk, "content") and chunk.content:
                    print(chunk.content, end="", flush=True)
    async def stream_query(self, question:str)-> AsyncIterator[str]:
        cache_res = self.cache.get(question)
        if cache_res["hit"]:
            yield cache_res["response"]
            return
            
        full_response = ""
        # On passe bien un dictionnaire avec la clé "question"
        async for chunk in self.rag_chain.astream(
            {"question": question},
            config = {"configurable": {"session_id": self.session_id}} 
        ):
            full_response += chunk
            yield chunk
            
        self.cache.set(question, full_response)
        
        # t'es jolie
     

                
            
        
            
            
            
        
            
            
        