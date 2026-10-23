from langchain_classic.memory import ConversationSummaryBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from config import settings
def build_local_memory(llm, max_token_limit : int=1000)-> ConversationSummaryBufferMemory:
    memory = ConversationSummaryBufferMemory(llm=llm,
                                             max_token_limit=max_token_limit,
                                             memory_key='chat_history',
                                             return_messages = True,
                                             ai_prefix='Tourism_AI',
                                             human_prefix = 'Tourist')
    return memory

from langchain_core.chat_history import InMemoryChatMessageHistory

store = {}
def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

def build_chain_with_history(chain) -> RunnableWithMessageHistory:
    """Enveloppe une chaîne LangChain existante avec la gestion de l'historique en mémoire."""
    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )