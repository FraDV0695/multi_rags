from typing import List

from ollama import ChatResponse

from dev.databases.vectordb_faiss import FAISSIndexManager
from dev.rags import RAG
from dev.utils.embedder import DocumentEmbedder
from dev.llm.ollama import Ollama
from dev.utils.entities import Document
from time import time

SYSTEM_PROMPT_VRAG = """
Sei un assistente virtuale che risponde alle domande utilizzando solo il contesto fornito. 
Per ogni risposta, segui queste istruzioni:

1. Genera una risposta che sia chiara, accurata e con una lunghezza approssimativa di 50-70 parole.
2. La risposta deve essere genrata solo sulla base del contesto fornito.
3. Se non sei sicuro o se il contesto non fornisce abbastanza informazioni, dì semplicemente "Non lo so."

Contesto:
{context_documents}

Rispondi alla domanda dell'utente fornendo:
Risposta: <<Risposta generata dal modello>>
"""

class VectorRAG(RAG):
    """Implementazione concreta di Retrieval-Augmented Generation (RAG) utilizzando un database vettoriale FAISS."""

    def __init__(self, index_path: str, metadata_path: str, llm_name: str, embedding_name:str):
        """
        Inizializza l'architettura RAG con un database vettoriale FAISS, un LLM e un embedder.

        :param index_path: Percorso dell'indice vettoriale FAISS.
        :param metadata_path: Percorso dei metadati associati ai documenti indicizzati.
        :param llm_name: Nome del LLM utilizzato per la generazione.
        :param embedding_name: Nome del modello di embedding per calcolare i vettori.
        """

        print("Caricamento dell'architettura......")
        start = time()
        self.embedder = DocumentEmbedder(model_name=embedding_name)
        self.index_manager = FAISSIndexManager(embedder=self.embedder, metric="L2", index_type="Flat")
        self.llm = Ollama(model_name=llm_name) #llama3.1:8b-instruct-q3_K_S
        self.index_path = index_path
        self.metadata_path = metadata_path
        print(f"Tempo impiegato per il caricamento dei componenti: {time() - start}\n")

        print("Caricamento del VectorDB.....")
        start = time()
        self.load_data()
        print(
            f"Tempo impiegato per la ricerca semantica dei chunk rilevanti rispetto la query fornita: {time() - start}\n")

    def load_data(self):
        """Carica l'indice FAISS e i metadati associati dal disco."""
        self.index_manager.load_from_disk(self.index_path, self.metadata_path)

    def search(self, query: str, top_k:int=5) -> List[Document]:
        """
        Esegue una ricerca semantica nel database vettoriale FAISS.

        :param query: Query testuale fornita dall'utente.
        :param top_k: Numero massimo di documenti da recuperare.
        :return: Lista di documenti rilevanti per la query.
        """
        print("Ricerca semantica nel VectorDB.....")
        start = time()
        embedding = self.embedder.embed_documents(
            [Document(text=query, metadata={})]
        ).documents[0].metadata['embedding']

        retreived_docs = self.index_manager.search(query_embedding=embedding, top_k=top_k)
        print(
            f"Tempo impiegato per la ricerca semantica dei chunk rilevanti rispetto la query fornita: {time() - start}\n")
        return retreived_docs


    def generate_answer(self, query: str, context: List[Document]) -> ChatResponse:
        """
        Genera una risposta basata sulla query dell'utente e sul contesto fornito.

        :param query: Query testuale dell'utente.
        :param context: Lista di documenti rilevanti recuperati durante la ricerca.
        :return: Risposta generata dal modello, inclusa di metadati.
        """
        print("Generazione della risposta alla query iniziale a partire dal contesto fornito.....")
        start = time()
        context_str = "\n".join([f"{doc.text}" for doc in context])
        prompt = f"{SYSTEM_PROMPT_VRAG}\nContesto:\n{context_str}"
        result = self.llm.generate(query=query, context=prompt)
        print(f"Tempo impiegato per la generazione della risposta d aparte del modello: {time() - start}\n")
        return result


if __name__ == "__main__":

    vrag = VectorRAG(
        index_path="../../database/faiss_index/documents.index",
        metadata_path="../../database/faiss_index/documents_metadata.parquet",
        llm_name="llama3.1:latest",
        embedding_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    query="In cosa consiste l'intelligenza artificiale?"
    retrieved_docs = vrag.search(query, top_k=10)

    generation = vrag.generate_answer(
        query="In cosa consiste l'intelligenza artificiale?", context=retrieved_docs
    )

    print(f"Risposta generata: {generation['message']['content']}")
    print(
        f"""Tempo impiegato dal modello per generare la risposta: {
        generation.total_duration / (10 ** 9) if generation.total_duration is not None else 'Non pervenuto'
        }""")