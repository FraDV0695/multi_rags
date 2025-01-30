import faiss
import numpy as np
from typing import List
from dev.utils.embedder import DocumentEmbedder
from dev.utils.entities import Document, DocumentDataset

class FAISSIndexManager:
    def __init__(self, embedder: DocumentEmbedder, metric: str = "L2", index_type: str = "Flat"):
        """
        Inizializza un gestore di indice FAISS.

        :param embedder: L'oggetto DocumentEmbedder per generare embeddings se necessario.
        :param metric: Metrica di distanza, "L2" (euclidea) o "IP" (Inner Product).
        :param index_type: Tipo di indice FAISS ("Flat", "IVFFlat", "HNSWFlat").
        """
        self.embedder = embedder
        self.dimension = self.embedder.model.get_sentence_embedding_dimension()
        self.metric = metric
        self.datastorage = DocumentDataset(documents=[])
        self.index = self._create_index(index_type)

    def _create_index(self, index_type: str):
        """
        Crea un indice FAISS in base al tipo specificato.
        """
        if index_type == "Flat":
            if self.metric == "L2":
                return faiss.IndexFlatL2(self.dimension)
            elif self.metric == "IP":
                return faiss.IndexFlatIP(self.dimension)
        elif index_type == "IVFFlat":
            quantizer = faiss.IndexFlatL2(self.dimension)  # Usiamo L2 come quantizer
            nlist = 100  # Numero di clusters
            index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_L2)
            index.train_required = True
            return index
        elif index_type == "HNSWFlat":
            index = faiss.IndexHNSWFlat(self.dimension, 32)  # 32 è un valore tipico per il parametro M
            return index
        else:
            raise ValueError("Tipo di indice non supportato")

    def add_documents(self, documents: DocumentDataset, embedding_key: str = "embedding"):
        """
        Aggiunge documenti con embedding all'indice FAISS.

        :param documents: Lista di oggetti Document.
        :param embedding_key: Nome della chiave nei metadati che contiene gli embedding.
        """
        embeddings = []
        docs_to_embed = []

        for doc in documents.documents:
            embedding = doc.metadata.get(embedding_key)
            if embedding is None:
                docs_to_embed.append(doc)
            else:
                embeddings.append(np.array(embedding, dtype='float32'))
                self.datastorage.add_document(doc)

        # Calcola gli embedding per i documenti senza embedding
        if docs_to_embed:
            embedded_docs = self.embedder.embed_documents(docs_to_embed)
            for doc in embedded_docs.documents:
                embedding = np.array(doc.metadata[embedding_key], dtype='float32')
                embeddings.append(embedding)
                self.datastorage.add_document(doc)

        # Aggiungi gli embedding all'indice FAISS
        if embeddings:
            embeddings = np.vstack(embeddings)  # Combina in una matrice
            if not self.index.is_trained:
                self.index.train(embeddings)
            self.index.add(embeddings)
            print(f"Aggiunti {len(embeddings)} documenti all'indice FAISS.")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Document]:
        """
        Esegue una ricerca sull'indice FAISS.

        :param query_embedding: Vettore embedding da cercare.
        :param top_k: Numero di risultati da restituire.
        :return: Lista di oggetti Document dei top_k risultati.
        """
        if self.index is None or self.index.ntotal == 0:
            raise ValueError("L'indice FAISS è vuoto o non inizializzato.")

        query_embedding = np.array(query_embedding, dtype="float32").reshape(1, -1)
        distances, indices = self.index.search(query_embedding, top_k)
        results = [self.datastorage.documents[idx] for idx in indices[0] if idx != -1]
        return results

    def save_to_disk(self, index_path: str, metadata_path: str):
        """
        Salva l'indice FAISS e i metadati associati ai documenti su disco.

        :param index_path: Percorso del file per salvare l'indice.
        :param metadata_path: Percorso del file per salvare i metadati dei documenti.
        """
        faiss.write_index(self.index, index_path)
        self.datastorage.to_parquet(metadata_path)
        print(f"Indice e metadati salvati rispettivamente in {index_path} e {metadata_path}.")

    def load_from_disk(self, index_path: str, metadata_path: str):
        """
        Carica l'indice FAISS e i metadati associati ai documenti da disco.

        :param index_path: Percorso del file dell'indice da caricare.
        :param metadata_path: Percorso del file dei metadati da caricare.
        """
        self.index = faiss.read_index(index_path)
        self.datastorage = DocumentDataset.from_parquet(metadata_path)
        print("Indice e metadati caricati con successo da disco.")