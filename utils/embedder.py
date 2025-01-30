from typing import List

import torch.cuda
from dev.utils.entities import Document, DocumentDataset
from sentence_transformers import SentenceTransformer


class DocumentEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-MiniLM-L3-v2"):
        """
        Inizializza il DocumentEmbedder con un modello di Sentence Transformers.

        :param model_name: Nome del modello di Sentence Transformers.
        """
        self.model = SentenceTransformer(
            model_name_or_path=model_name,
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )

    def embed_documents(self, documents: List[Document], batch_size: int = 32) -> DocumentDataset:
        """
        Genera embeddings per una lista di documenti LangChain.

        :param documents: Lista di oggetti Document.
        :param batch_size: Elabora (encode) le frasi con un batch di dimensioni specificate. (default: 32)
        :return: Lista di oggetti Document aggiornati con embeddings.
        """
        # Assicurarsi che la lista non sia vuota
        if not documents:
            raise ValueError("La lista dei documenti è vuota.")

        embeddings = self.model.encode(sentences=[doc.text for doc in documents], batch_size=batch_size, show_progress_bar=True)
        docs_w_emb = [
            Document(
                text=doc.text,
                metadata={
                    **doc.metadata.copy(), 'embedding': embedding
                }
            )
            for doc, embedding in zip(documents, embeddings)
        ]
        return DocumentDataset(documents=docs_w_emb)
