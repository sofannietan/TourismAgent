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

# Après les imports LangChain (ou ici si l'import ci-dessous fonctionne)
from langchain_core._api import LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
os.environ["OPENBLAS_NUM_THREADS"]='1'
os.environ["MKL_NUM_THREADS"]="1"

# Importez vos modules (ajustez selon votre structure de projet)
from src.chains.rag_chain import TourismAgentPipeline
from src.agents.agent import router_query

console = Console()

def display_welcome_banner():
    """Affiche la bannière de bienvenue."""
    welcome_text = "Hi ! I'm KhmerAgent, an agent specialized in tourism in Cambodia"
    console.print(Panel(welcome_text, title="[bold gold1]KhmerAgent [/bold gold1]", border_style="gold1"))

async def select_mode() -> str:
    """Permet à l'utilisateur de choisir entre mode RAG, Agent ou Auto."""
    console.print("\n[bold cyan]Choisissez un mode d'exécution :[/bold cyan]")
    console.print(" [1] [green]auto[/green]   - Routage automatique (décide entre RAG et Agent)")
    console.print(" [2] [green]rag[/green]    - Chaîne RAG classique avec mémoire conversationnelle")
    console.print(" [3] [green]agent[/green]  - Agent ReAct multi-outils (avec streaming Tool Calls)")

    choice = Prompt.ask(
        "\nVotre choix",
        choices=["1", "2", "3", "auto", "rag", "agent"],
        default="auto"
    )

    mapping = {
        "1": "auto",
        "2": "rag",
        "3": "agent",
    }
    
    selected = mapping.get(choice, choice)
    console.print(f"[bold green]✓ Mode sélectionné : {selected}[/bold green]\n")
    return selected
async def handle_streaming_query(pipeline, question: str):
    """Exécute et affiche la réponse RAG en streaming token par token."""
    console.print("\n[bold blue] Maison IQ :[/bold blue] ", end="")
    try:
        # FIX 1 : Utilisation de print() standard au lieu de console.print pour flush=True
        async for chunk in pipeline.stream_query(question):
            print(chunk, end="", flush=True)
        print("\n")
    except Exception as e:
        console.print(f"\n[bold red]❌ Erreur lors du streaming : {e}[/bold red]")

def handle_tourist_query(pipeline, question: str):
    """Exécute la requête et affiche les informations du TouristAgent formatées dans un tableau."""
    console.print("\n[bold blue]🧭 Génération de la réponse touristique...[/bold blue]")
    try:
        # Adaptez selon la méthode de votre pipeline pour le TouristAgent
        result = pipeline.query(question, mode="tourist")

        table = Table(title="Réponse TouristAgent Maison IQ", border_style="gold1", show_header=True)
        table.add_column("Champ", style="bold cyan", width=18)
        table.add_column("Détails", style="white")

        if hasattr(result, "answer"):
            table.add_row("Réponse", result.answer)
        elif isinstance(result, dict) and "answer" in result:
            table.add_row("Réponse", str(result["answer"]))
        else:
            table.add_row("Résultat", str(result))

        if hasattr(result, "sources") and result.sources:
            table.add_row("Sources", ", ".join(result.sources) if isinstance(result.sources, list) else str(result.sources))

        if hasattr(result, "confidence_score") and result.confidence_score is not None:
            table.add_row("Confiance", f"{result.confidence_score * 100:.1f}%")

        if hasattr(result, "follow_up_questions") and result.follow_up_questions:
            questions_str = "\n".join([f"• {q}" for q in result.follow_up_questions])
            table.add_row("Questions de suivi", questions_str)

        console.print(table)
        console.print("\n")
    except Exception as e:
        console.print(f"[bold red]❌ Erreur lors de la requête touristique : {e}[/bold red]")

async def main_loop():
    display_welcome_banner()
    
    session_id = f"session_{uuid.uuid4().hex[:8]}"
    console.print(f"[dim]ID de Session attribué : {session_id}[/dim]")
    console.print("[yellow]Chargement du pipeline Maison IQ (Embeddings & VectorStore)...[/yellow]")

    # Initialisation de votre pipeline (remplacez par votre classe)
    pipeline =TourismAgentPipeline(session_id=session_id)
    console.print("[bold green]✓ Pipeline initialisé et prêt ![/bold green]\n")
    current_mode = await select_mode()
    

    while True:
        try:
            console.print(Rule(style="dim"))
            user_input = Prompt.ask(f"[bold gold1]Maison IQ ({current_mode})[/bold gold1] >").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                console.print("\n[bold red]Fermeture de Maison IQ. Au revoir ![/bold red]")
                break

            if user_input.lower() == "/mode":
                current_mode = select_mode()
                continue

            if user_input.lower() == "/session":
                console.print(f"[bold cyan]ID de session actif :[/bold cyan] {session_id}\n")
                continue

                
            # ... dans le while True, après la vérification des commandes spéciales (/mode, /session, etc.) :

            if current_mode == "agent":
                await pipeline.stream_agent_events(user_input)
                
            elif current_mode == "rag":
                await handle_streaming_query(pipeline, user_input)
                
            elif current_mode == "auto":
                mode_decide = router_query(user_input, pipeline.router)
                console.print(f"[dim]⚡ Routeur : Mode [{mode_decide}] sélectionné[/dim]")
                
                if mode_decide == "agent":
                    await pipeline.stream_agent_events(user_input)
                else:
                    await handle_streaming_query(pipeline, user_input)
            else:
                await handle_streaming_query(pipeline, user_input)

        except KeyboardInterrupt:
            console.print("\n[bold red]Interruption détectée. Fermeture...[/bold red]")
            break
        except Exception as e:
            console.print(f"\n[bold red]Une erreur est survenue : {e}[/bold red]")

if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        console.print("\n[bold red]Fermeture de Maison IQ. Au revoir ![/bold red]")
        sys.exit(0)