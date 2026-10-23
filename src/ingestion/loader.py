from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader

def load_documents(data_dir: str) -> list:
    file_extension = Path(data_dir).suffix.lower()
    
    if file_extension == ".pdf":
        loader = PyPDFLoader(file_path=data_dir)
        return loader.load()
        
    elif file_extension == ".txt":
        # On essaie d'abord utf-8, et si ça plante (UnicodeDecodeError), on essaie latin-1 (très fréquent sous Windows)
        try:
            loader = TextLoader(file_path=data_dir, encoding="utf-8")
            return loader.load()
        except UnicodeDecodeError:
            loader = TextLoader(file_path=data_dir, encoding="latin-1")
            return loader.load()
            
    else:
        raise ValueError(f"File type unsupported ({file_extension}). Must be a .txt or a .pdf")