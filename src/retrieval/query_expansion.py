from langchain_classic.retrievers import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def build_multiquery_retriever(vectorstore,llm, k:int=5)-> MultiQueryRetriever:
    base_retriever = vectorstore.as_retriever(search_kwargs={"k":k})
    multiquery_retriever = MultiQueryRetriever.from_llm(retriever=base_retriever,llm=llm)
    return multiquery_retriever

def deduplicate_documents(documents:list)-> list:
    seen_contents = set()
    unique_documents=[]
    for doc in documents :
        if doc.page_content not in seen_contents:
            seen_contents.add(doc.page_content)
            unique_documents.append(doc)
    return unique_documents

def build_hyde_prompt()-> ChatPromptTemplate:
    """Construit un prompt template pour la méthode HyDE. Celui ci demande au LLM de générer une réponse fictive à la question."""
    system_prompt =(
        "Vous êtes un expert en rédaction de documents et rapports.\n"
        "Votre tâche est de rédiger une réponse fictive au prompt ou un extrait de document hypothétique qui répondrait parfaitement et de manière détaillée à la question posée \n"
        "Consignes: \n"
        "-Rédigez le texte directement comme s'il s'agissait d'un extrait de rapport officiel ou un article. \n"
        "-Ne mettez pas de formule de politesse ou d'introduction comme 'Voici un document...' \n"
        "-Utilisez un vocabulaire technique et professionnel adapté au domaine \n"
    )
    prompt = ChatPromptTemplate.from_messages([("system",system_prompt), ("human"),"Question de l'utilisateur: {question} \n\n Document hypothétique idéal :"])
    return prompt

def hyde_retrieve(question, llm, vectorstore, k:int=5)-> list:
    prompt=build_hyde_prompt()
    hyde_chain = prompt | llm| StrOutputParser()
    hypothetical_doc = hyde_chain.invoke({"question":question})
    retriever= vectorstore.as_retriever(search_kwargs={"k":k})
    docs = retriever.invoke(hypothetical_doc)
    return docs

def hyde_multiquery_retrieve(question, llm, vectorstore, k:int=5)-> list:
    """Combine HyDE et MultiQueryRetriever:
    1. Génère un document hypothétique
    2. Utilise le MultiQueryRetriever sur ce doc hypothétique
    3. Déduplique les résultats"""
    prompt=build_hyde_prompt()
    hyde_chain = prompt | llm| StrOutputParser()
    hypothetical_doc = hyde_chain.invoke({"question":question})
    mq_retriever= build_multiquery_retriever(vectorstore=vectorstore, llm = llm, k=k)
    docs = mq_retriever.invoke(hypothetical_doc)
    unique_docs = deduplicate_documents(docs)
    return unique_docs

    