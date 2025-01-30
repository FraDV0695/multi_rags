from typing import List

from dev.utils.entities import Document, DocumentDataset
import re


class DocumentChunker:

    @staticmethod
    def _chunk_by_word(text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Divide il testo in chunk di lunghezza specificata con sovrapposizione.

        :param text: Il testo da suddividere.
        :param chunk_size: Numero massimo di caratteri per chunk.
        :param overlap: Numero di caratteri sovrapposti tra chunk consecutivi.
        :return: Lista di chunk.
        """
        words = text.split()
        chunks = []
        start = 0

        while start < len(words):
            # Calcola l'indice di fine del chunk
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            # Aggiorna l'inizio del prossimo chunk tenendo conto dell'overlap
            start += chunk_size - overlap

        return chunks

    @staticmethod
    def _chunk_by_sentence(text: str, max_sentences: int) -> List[str]:
        """
        Divide il testo in chunk basati su frasi.

        :param text: Il testo da suddividere.
        :param max_sentences: Numero massimo di frasi per chunk.
        :return: Lista di chunk.
        """
        # Divide il testo in frasi basandosi sulla punteggiatura
        sentences = re.split(r'(?<=[.!?]) +', text.strip())
        chunks = []
        current_chunk = []

        for sentence in sentences:
            current_chunk.append(sentence)
            if len(current_chunk) >= max_sentences:
                chunks.append(" ".join(current_chunk))
                current_chunk = []

        # Aggiungi eventuali frasi rimanenti
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def generate_chunks(self,
                        documents: List[Document],
                        method: str = 'word',
                        chunk_size: int = None,
                        overlap: int = None,
                        max_sentences: int = None) -> DocumentDataset:
        """
        Genera chunk da una lista di oggetti Document con metodi diversi.

        :param documents: Lista di oggetti Document da suddividere in chunk.
        :param method: Metodo di chunking ('word' o 'sentence').
        :param chunk_size: Numero massimo di parole per chunk (usato con method='word').
        :param overlap: Numero di parole sovrapposte tra chunk consecutivi (usato con method='word').
        :param max_sentences: Numero massimo di frasi per chunk (usato con method='sentence').
        :return: DocumentDataset con i chunk.
        """

        supported_methods = ['word', 'sentence']

        # Verifica che il metodo sia supportato
        if method not in supported_methods:
            raise ValueError(
                f"Metodo di chunking non supportato: '{method}'. "
                f"I metodi disponibili sono: {', '.join(supported_methods)}."
            )

        if method == 'word' and (chunk_size is None or overlap is None):
            raise ValueError("I parametri 'chunk_size' e 'overlap' devono essere specificati per il metodo 'word'.")
        elif method == 'sentence' and max_sentences is None:
            raise ValueError("Il parametro 'max_sentences' deve essere specificato per il metodo 'sentence'.")

        chunked_documents = []
        for doc in documents:
            if method == 'word':
                chunks = self._chunk_by_word(doc.text, chunk_size, overlap)
            elif method == 'sentence':
                chunks = self._chunk_by_sentence(doc.text, max_sentences)
            else:
                raise ValueError(f"Metodo di chunking non supportato: {method}")

            for i, chunk in enumerate(chunks):
                chunked_documents.append(
                    Document(
                        text=chunk,
                        metadata={
                            **doc.metadata,
                            "id_chunk": f"{doc.metadata.get('id', 'unknown')}_{i + 1}"
                        }
                    )
                )

        return DocumentDataset(documents=chunked_documents)