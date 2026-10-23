from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Chemin absolu vers la racine du projet où se trouve le .env
BASE_DIR = Path(__file__).resolve().parent
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Chemin absolu vers la racine du projet où se trouve le .env
BASE_DIR = Path(__file__).resolve().parent

class Settings(BaseSettings):
    model_name: str = "gemini-3.1-flash-lite"
    embedding_model: str = "gemini-embedding-001"
    location: str = "europe-west1"
    
    GOOGLE_API_KEY: str
    
    chunk_size: int = 300
    chunk_overlap: int = 50
    top_k_retrieval: int = 5
    
    dossier_wiki: Path = BASE_DIR / "data" / "wiki_pages"

    @property
    def file_paths(self) -> list[str]:
        if self.dossier_wiki.exists():
            txt = [str(file.resolve()) for file in self.dossier_wiki.glob("*.txt")]
            pdf = [str(file.resolve()) for file in self.dossier_wiki.glob("*.pdf")]
            return txt + pdf
        return []

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",  # Fixe le chemin absolu vers .env
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()