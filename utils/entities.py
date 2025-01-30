from typing import Dict, Any, List, Optional
import pandas as pd

from pydantic import BaseModel

class Document(BaseModel):
    """Rappresentazione generale di un documento, caratterizzato da un contenuto (text) e da dei dati aggiuntivi da inserire in metadata."""
    text: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """
        Converte l'oggetto Document in un dizionario.

        :return: Dizionario con le chiavi 'text' e 'metadata'.
        """
        return {"text": self.text, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """
        Crea un oggetto Document da un dizionario.

        :param data: Dizionario con le chiavi 'text' e 'metadata'.
        :return: Oggetto Document creato dal dizionario.
        """
        if "text" not in data or "metadata" not in data:
            raise ValueError("Il dizionario deve contenere le chiavi 'text' e 'metadata'.")
        return cls(text=data["text"], metadata=data["metadata"])

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Document):
            return False
        return self.text == other.text and self.metadata == other.metadata


class DocumentDataset(BaseModel):
    """Wrapper per dataset composti da oggetti Document"""
    documents: List[Document]

    def add_document(self, document: Document) -> None:
        """
        Aggiunge un nuovo oggetto Document al dataset.
        """
        self.documents.append(document)

    def filter_by_metadata(self, key: str, value: Any) -> "DocumentDataset":
        """
        Filtra i documenti in base a una chiave e un valore specificati nei metadati.
        """
        filtered_documents = [doc for doc in self.documents if doc.metadata.get(key) == value]
        return DocumentDataset(documents=filtered_documents)

    def get_all_texts(self) -> List[str]:
        """
        Restituisce una lista di tutti i testi presenti nei documenti.
        """
        return [doc.text for doc in self.documents]

    def get_metadata_values(self, key: str) -> List[Any]:
        """
        Restituisce tutti i valori associati a una chiave specifica nei metadati.
        """
        return [doc.metadata.get(key) for doc in self.documents if key in doc.metadata]

    def remove_document_by_metadata(self, key: str, value: Any) -> None:
        """
        Rimuove tutti i documenti che corrispondono a una chiave e un valore specificati nei metadati.
        """
        self.documents = [doc for doc in self.documents if doc.metadata.get(key) != value]

    def to_json(self) -> str:
        """
        Converte l'intero dataset in formato JSON.
        """
        return self.model_dump_json()

    @classmethod
    def from_json(cls, json_str: str) -> "DocumentDataset":
        """
        Crea un oggetto DocumentDataset da una stringa JSON.
        """
        return cls.model_validate_json(json_str)

    def to_csv(self, file_path: str) -> None:
        """
        Esporta i documenti in un file CSV.
        """
        data = [doc.to_dict() for doc in self.documents]
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)

    def to_parquet(self, file_path: str) -> None:
        """
        Esporta i documenti in un file Parquet.
        """
        data = [doc.to_dict() for doc in self.documents]
        df = pd.DataFrame(data)
        df.to_parquet(file_path, index=False)

    @classmethod
    def from_csv(cls, file_path: str) -> "DocumentDataset":
        """
        Crea un DocumentDataset da un file CSV.
        """
        # Leggere il CSV in un DataFrame
        df = pd.read_csv(file_path)

        # Estrarre la colonna `text` e i metadati
        texts = df["text"].tolist()
        metadata_columns = [col for col in df.columns if col != "text"]
        metadata = df[metadata_columns].to_dict(orient="records")

        # Creare i documenti in modo vettorizzato
        documents = [Document(text=text, metadata=meta) for text, meta in zip(texts, metadata)]
        return cls(documents=documents)

    @classmethod
    def from_parquet(cls, file_path: str) -> "DocumentDataset":
        """
        Crea un DocumentDataset da un file CSV.
        """
        # Leggere il CSV in un DataFrame
        df = pd.read_parquet(file_path)

        # Estrarre la colonna `text` e i metadati
        texts = df["text"].tolist()
        metadata_columns = [col for col in df.columns if col != "text"]
        metadata = df[metadata_columns].to_dict(orient="records")

        # Creare i documenti in modo vettorizzato
        documents = [Document(text=text, metadata=meta['metadata']) for text, meta in zip(texts, metadata)]
        return cls(documents=documents)

class Entity(BaseModel):
    """Rappresenta un'entità estratta dal testo."""
    text: str  # Testo dell'entità
    label: str  # Tipo dell'entità (es. PERSON, ORG, LOC)

class Relation(BaseModel):
    """Rappresenta una relazione estratta tra due entità."""
    subject: str  # Entità soggetto della relazione
    predicate: str  # Predicato originale (verbo/predicato completo)
    predicate_lemma: str  # Lemma del predicato
    tense: Optional[str] = "Sconosciuto"  # Tempo verbale
    object: str  # Entità oggetto della relazione
    url: Optional[str] = None  # URL o fonte della relazione

class QAData(BaseModel):
    """Modello che rappresenta una domanda e la relativa risposta."""
    question: str
    """La domanda da porre."""
    answer: str
    """La risposta corrispondente alla domanda."""

class QADataset(BaseModel):
    """Modello per il dataset di test composto da istanze di QAData."""
    qa: List[QAData]
    """Lista di domande e risposte."""

    def to_csv(self, file_path: str) -> None:
        """
        Esporta i documenti in un file CSV.
        """
        data = [elem.to_dict() for elem in self.qa]
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)

    @classmethod
    def from_csv(cls, file_path: str) -> "QADataset":
        """
        Crea un DocumentDataset da un file CSV.
        """
        # Leggere il CSV in un DataFrame
        df = pd.read_csv(file_path)

        # Estrarre la colonna `question` e answer
        questions = df["text"].tolist()
        answers = df['answers'].to_list()

        # Creare i documenti in modo vettorizzato
        qa_list = [QAData(question=text, answer=meta) for text, meta in zip(questions, answers)]
        return cls(qa=qa_list)