from abc import abstractmethod, ABC

import pandas as pd
import os
from typing import List

from tqdm import tqdm
from dev.utils.entities import Document


class DataManager(ABC):

    def __init__(self, ref_dir:str):
        """
        Inizializza la classe DataRawManager e crea la directory di salvataggio se non esiste.
        :param ref_dir: Percorso in cui salvare i dati scaricati
        """
        self.ref_dir = ref_dir
        if not os.path.exists(self.ref_dir):
            os.makedirs(self.ref_dir)

    @abstractmethod
    def load_data(self, data_name: str, text_column:str) -> List[Document]:
        """
        Carica il dataset salvato localmente in formato csv e lo trasforma in una lista di oggetti Document.
        :param data_name: Nome del file in cui recuperare il dati
        :param text_column: Nome della colonna in cui si trova il contenuto testuale del dato
        :return: Lista di oggetti Document
        """
        ...

    @abstractmethod
    def save_data(self, documents: List[Document], data_name: str):
        """
        Salva la lista di oggetti Document localmente con il nome fornito in data_name.
        :param documents: Collezione di oggetti Document da salvare
        :param data_name: Nome del file in cui salvare i dati
        :return: True se il file è stato salvato con successo, False altrimenti
        """
        ...


class DataCSVManager(DataManager):

    def load_data(self, data_name: str, text_column:str) -> List[Document]:
        """
        Carica il dataset salvato localmente in formato csv e lo trasforma in una lista di oggetti Document.
        :param data_name: Nome del file in cui recuperare il dati
        :param text_column: Nome della colonna in cui si trova il contenuto testuale del dato
        :return: Lista di oggetti Document
        """
        # Carica il dataset salvato in precedenza

        dataset = pd.read_csv(f"{self.ref_dir}/{data_name}")
        bad_columns = [col for col in dataset.columns if "Unnamed:" in col]
        dataset = dataset.drop(bad_columns, axis=1)
        # Converti i dati nel formato Document di LangChain
        documents = [
            Document(
                text=getattr(row, text_column),  # Testo dell'articolo
                metadata={col: getattr(row, col) for col in dataset.columns if col != text_column}
            )
            for row in tqdm(dataset.itertuples(index=False), desc="Conversione dei dati raw in lista di Document")
        ]

        print(f"Caricati {len(documents)} documenti.")
        return documents

    def save_data(self, documents: List[Document], data_name: str) -> bool:
        """
        Salva la lista di oggetti Document localmente con il nome fornito in data_name.
        :param documents: Collezione di oggetti Document da salvare
        :param data_name: Nome del file in cui salvare i dati
        :return: True se il file è stato salvato con successo, False altrimenti
        """

        data = [{"text": doc.text, **doc.metadata} for doc in documents]

        df = pd.DataFrame(data)
        try:
            df.to_csv(f"{self.ref_dir}/{data_name}", index=False)
            return True
        except Exception as e:
            print(f"Errore in fase di salvataggio dei dati in file {self.ref_dir}/{data_name}:\n {e}")
            return False


# Utilizzo della classe
if __name__ == "__main__":
    manager = DataCSVManager(ref_dir="../../data")

    # Carica i dati dal disco e li converte in Document di LangChain
    documents = manager.load_data(data_name="subset_dataset_gcp_cleaned.csv", text_column='text')

    # Esempio di visualizzazione dei primi documenti
    for doc in documents[:5]:
        print(f"Titolo: {doc.metadata['title']}")
        print(f"Testo: {doc.text[:500]}...")  # Stampa i primi 500 caratteri del testo
        print()