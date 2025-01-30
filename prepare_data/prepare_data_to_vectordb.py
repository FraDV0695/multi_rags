from dev.utils.chunker import DocumentChunker
from dev.utils.data_manager import DataCSVManager
from dev.prepare_data.clean_dataset import clean_documents
from dev.utils.embedder import DocumentEmbedder
from dev.databases.vectordb_faiss import FAISSIndexManager


def process_batch(documents, embedder, index_manager, batch_size=1000):
    # Pulizia dei testi nei documenti
    documents = clean_documents(documents)

    # Generazione dei chunk di documenti
    documents = chunker.generate_chunks(documents, chunk_size=200, overlap=20)

    # Calcolo dell'embedding per chunk di documenti
    for start in range(0, len(documents.documents), batch_size):
        batch_docs = documents.documents[start:start + batch_size]
        batch_docs = embedder.embed_documents(batch_docs, batch_size=16)

        # Aggiungi documenti con embedding all'indice FAISS
        index_manager.add_documents(batch_docs, embedding_key="embedding")

if __name__ == "__main__":
    data_manager = DataCSVManager(ref_dir="../../data")
    chunker = DocumentChunker()
    embedder = DocumentEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index_manager = FAISSIndexManager(embedder=embedder, metric="L2", index_type="IVFFlat")


    documents = data_manager.load_data(data_name="subset_dataset_gcp_cleaned.csv", text_column='text')

    # Partizionamento e processo dei dati in batch
    batch_size = 1000  # Modifica questa dimensione in base alle tue esigenze di memoria e alla dimensione del dataset
    for start in range(0, len(documents), batch_size):
        batch_docs = documents[start:start + batch_size]
        process_batch(batch_docs, embedder, index_manager, batch_size)

    # Salva l'indice FAISS e i metadati dei documenti
    index_manager.save_to_disk(
        index_path="../../database/faiss_index/documents.index",
        metadata_path="../../database/faiss_index/documents_metadata.parquet"
    )

    print('Fatto')