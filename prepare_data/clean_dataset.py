from typing import List
import re
from tqdm import tqdm

from dev.utils.data_manager import DataCSVManager
from dev.utils.entities import Document


def clean_documents(documents: List[Document]) -> List[Document]:
    """
    Pulisce i testi nella lista di oggetti Document.

    :param documents: Lista di oggetti Document da pulire.
    :return: Lista di oggetti Document con testi puliti.
    """
    # Compilazione delle regex per migliorare le prestazioni
    newline_tab_pattern = re.compile(r'[\n\t]')
    multiple_spaces_pattern = re.compile(r'\s+')
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    list_pattern = re.compile(r'\d+\.\s*|- ')
    references_pattern = re.compile(r'\[\S+\]')

    # Funzione per pulire un singolo testo
    def clean_text(text: str) -> str:
        text = newline_tab_pattern.sub(' ', text)
        text = multiple_spaces_pattern.sub(' ', text)
        text = url_pattern.sub('', text)
        text = list_pattern.sub('', text)
        text = references_pattern.sub('', text)
        return text.strip()

    # Applicare la pulizia a ciascun Document
    for doc in tqdm(documents, desc="Pulizia in corso"):
        doc.text = clean_text(doc.text)

    return documents

if __name__ == "__main__":
    # Esempio di utilizzo
    manager = DataCSVManager(ref_dir="../../data")

    # Carica i dati dal disco e li converte in Document di LangChain
    documents = manager.load_data(data_name="subset_dataset_gcp_cleaned.csv", text_column='text')

    # Pulizia della colonna 'text'
    data_cleaned = clean_documents(documents)

    # Salva o visualizza i risultati
    result = manager.save_data(data_cleaned, 'subset_dataset_cleaned.csv')
    print("Dati salvati con successo" if result else "C'è stato quale problema")
