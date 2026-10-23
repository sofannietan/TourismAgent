from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunking(documents:list,chunk_size:int, chunk_overlap:int)-> list:
    
    splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs=[]
    if len(documents)> 100:
        batch_size= 100
        for i in range(0, len(documents), batch_size):
            batch_doc = documents[i:i+batch_size]
            split_docs.extend(splitter.split_documents(batch_doc))
    else :
        split_docs.extend(splitter.split_documents(documents))
    return split_docs



    