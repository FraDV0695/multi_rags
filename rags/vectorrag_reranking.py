from typing import List
from dev.rags.vectorrag import VectorRAG
from dev.utils.entities import Document
from sentence_transformers import CrossEncoder

SYSTEM_PROMPT_VRAG = """
Sei un assistente virtuale che risponde alle domande utilizzando solo il contesto fornito. 
Il contesto è composto da estratti di documenti, di cui dovrai considerare esclusivamente il contenuto del campo 'text'. 
Per ogni risposta, segui queste istruzioni:

1. Cita direttamente il testo rilevante come prova per la tua risposta.
2. Fornisci la fonte della citazione, includendo metadati come l'id del chunk e il titolo del documento.
3. Sii il più conciso possibile nella tua risposta.

Se non sei sicuro o se il contesto non fornisce abbastanza informazioni, dì semplicemente "Non lo so."

Contesto:
{context_documents}

Rispondi alla domanda dell'utente fornendo:
Risposta: <<Risposta generata dal modello>>
"""

class VectorRAGReranking(VectorRAG):
    """Implementazione estesa di Retrieval-Augmented Generation (RAG) con un meccanismo di reranking."""

    def __init__(self, index_path: str, metadata_path: str, llm_name: str, embedding_name: str,  cross_encoder_model: str):
        """
        Inizializza l'architettura RAG con reranking utilizzando un cross-encoder.

        :param index_path: Percorso dell'indice vettoriale FAISS.
        :param metadata_path: Percorso dei metadati associati ai documenti indicizzati.
        :param llm_name: Nome del LLM utilizzato per la generazione.
        :param embedding_name: Nome del modello di embedding per calcolare i vettori.
        :param cross_encoder_model: Nome del modello cross-encoder per il reranking.
        """
        super().__init__(index_path, metadata_path, llm_name, embedding_name)
        self.cross_encoder = CrossEncoder(cross_encoder_model)

    def search(self, query: str, top_k:int=5) -> List[Document]:
        """
        Esegue una ricerca semantica nel database vettoriale FAISS e applica il reranking.

        :param query: Query testuale fornita dall'utente.
        :param top_k: Numero massimo di documenti da restituire dopo il reranking.
        :return: Lista di documenti rilevanti dopo il reranking.
        """
        results = super().search(query=query, top_k=top_k*2)
        scores = self.cross_encoder.predict([[query, doc.text] for doc in results])
        return [doc for doc, _ in sorted(zip(results, scores), key=lambda pair: pair[1], reverse=True)][:top_k]

if __name__ == "__main__":

    vrag_rk = VectorRAGReranking(
        index_path="../../database/faiss_index/documents.index",
        metadata_path="../../database/faiss_index/documents_metadata.parquet",
        llm_name="llama3.1:latest",
        embedding_name="sentence-transformers/all-MiniLM-L6-v2",
        cross_encoder_model="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    )

    query = "In cosa consiste l'intelligenza artificiale?"
    retrieved_docs = vrag_rk.search(query, top_k=10)

    generation = vrag_rk.generate_answer(
        query="In cosa consiste l'intelligenza artificiale?", context=retrieved_docs
    )

    print(f"Risposta generata: {generation['message']['content']}")
    print(
        f"""Tempo impiegato dal modello per generare la risposta: {
        generation.total_duration / (10 ** 9) if generation.total_duration is not None else 'Non pervenuto'
        }""")