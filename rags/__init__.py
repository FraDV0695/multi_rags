from abc import ABC, abstractmethod
from typing import List
from ollama import ChatResponse
from dev.utils.entities import Document


class RAG(ABC):
    """
    Classe astratta per implementazioni di Retrieval-Augmented Generation (RAG).

    Metodi astratti:
        - load_data: Carica i dati da una sorgente esterna.
        - search: Cerca i documenti rilevanti rispetto a una query.
        - generate_answer: Genera una risposta basata sui documenti recuperati.
    """

    @abstractmethod
    def load_data(self, **kwargs) -> None:
        """Carica i dati richiesti per la ricerca."""
        pass

    @abstractmethod
    def search(self, query: str, **kwargs) -> List[Document]:
        """
        Esegue la ricerca dei documenti rilevanti per una query specifica.
        :param query: La query di ricerca formulata dall'utente.
        :param kwargs: Argomenti opzionali per configurare la ricerca (es. numero massimo di risultati).
        :return: Una lista di documenti rilevanti.
        """
        pass

    @abstractmethod
    def generate_answer(self, query: str, context: List[Document]) -> ChatResponse:
        """
        Genera una risposta basata sulla query e sul contesto fornito.

        :param query: La query di input dell'utente.
        :param context: Un elenco di documenti recuperati dalla ricerca.
        :return: La risposta generata sotto forma di oggetto ChatResponse.
        """
        pass
