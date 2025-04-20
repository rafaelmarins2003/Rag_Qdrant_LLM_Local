from typing import List
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory

# 1) Configuração da memória via pydantic
class MemoryConfig(BaseModel):
    max_messages: int = Field(
        10,
        description="Número máximo de mensagens a manter na memória"
    )

    class Config:
        # permite tipos arbitrários (nosso History é um BaseChatMessageHistory)
        arbitrary_types_allowed = True

# 2) Implementação concreta de BaseChatMessageHistory
class SimpleChatHistory(BaseChatMessageHistory):
    messages: List[BaseMessage]

    def __init__(self):
        super().__init__()
        self.messages = []

    def add_message(self, message: BaseMessage) -> None:
        """Adiciona uma mensagem ao histórico."""
        self.messages.append(message)

    def get_messages(self) -> List[BaseMessage]:
        """Retorna todas as mensagens armazenadas."""
        return self.messages

    def clear(self) -> None:
        self.messages = []

# 3) Classe de memória que une history + config
class ChatMemory(BaseModel):
    history: SimpleChatHistory
    config: MemoryConfig

    class Config:
        arbitrary_types_allowed = True

    def save_context(self, user_msg: HumanMessage, ai_msg: AIMessage):
        # adiciona as duas mensagens
        self.history.add_message(user_msg)
        self.history.add_message(ai_msg)
        # poda histórico se exceder max_messages
        if len(self.history.messages) > self.config.max_messages:
            self.history.messages = self.history.messages[-self.config.max_messages :]

    def load_memory(self) -> List[BaseMessage]:
        return self.history.get_messages()

# 4) Exemplo de uso
# if __name__ == "__main__":
#     # configura memória p/ até 4 mensagens
#     cfg = MemoryConfig(max_messages=4)
#     hist = SimpleChatHistory()
#     mem  = ChatMemory(history=hist, config=cfg)
#
#     # simula algumas interações
#     mem.save_context(
#         HumanMessage(content="Olá, quem é você?"),
#         AIMessage(content="Eu sou um assistente baseado em LangChain.")
#     )
#     mem.save_context(
#         HumanMessage(content="O que pode fazer?"),
#         AIMessage(content="Posso lembrar contexto e responder perguntas.")
#     )
#
#     # carrega e imprime memória atual
#     for msg in mem.load_memory():
#         role = "Usuário" if isinstance(msg, HumanMessage) else "Assistente"
#         print(f"{role}: {msg.content}")
