from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate

def build_conversational_chain(llm, retriever,memory)-> ConversationalRetrievalChain:
    conversational_chain = ConversationalRetrievalChain(
        llm = llm,
        retriever = retriever,
        memory = memory,
        return_source_documents=True,
        verbose = True,
        combine_docs_chain_kwargs={
            "prompt": ChatPromptTemplate.from_messages([
                ("system","""Vous êtes un agent IA spécialisé dans le tourisme au Cambodge. Utilisez l'historique de conversation et le contexte fourni pour répondre de manière cohérente et perminente.
                 Historique : {chat_history}
                 Contexte documentaire :{context}
                 """),
                ("human", "{question}")
                
            ])
        }
    )
    return conversational_chain

