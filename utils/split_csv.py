import pandas as pd
import os


def split_csv(input_file, output_prefix, chunk_size=100000, output_dir='output_chunks'):
    """
    Suddivide un file CSV in blocchi di dimensione specificata.

    :param input_file: Percorso del file CSV di input.
    :param output_prefix: Prefisso per i file CSV di output.
    :param chunk_size: Numero di righe per ogni blocco.
    :param output_dir: Directory dove salvare i file divisi.
    """
    # Crea la directory di output se non esiste
    os.makedirs(output_dir, exist_ok=True)

    # Inizializza il contatore dei blocchi
    chunk_number = 1

    # Leggi il CSV in chunk
    try:
        for chunk in pd.read_csv(input_file, chunksize=chunk_size, dtype=str):
            # Definisci il nome del file di output
            output_file = os.path.join(output_dir, f"{output_prefix}_part{chunk_number}.csv")

            # Salva il chunk nel file di output
            chunk.to_csv(output_file, index=False)

            print(f"Salvato: {output_file} con {len(chunk)} righe.")

            # Incrementa il contatore
            chunk_number += 1
    except pd.errors.EmptyDataError:
        print("Il file CSV è vuoto.")
    except pd.errors.ParserError as e:
        print(f"Errore durante il parsing del CSV: {e}")
    except Exception as e:
        print(f"Errore inaspettato: {e}")


if __name__ == "__main__":
    # Specifica il percorso del file CSV di input
    input_csv = '../data/output_neo4j/relazioni_filtrate.csv'  # Modifica questo percorso se necessario

    # Specifica il prefisso per i file di output
    output_pref = 'relazioni_filtrate'

    # Specifica la directory di output
    output_directory = '../data/output_neo4j'  # Puoi cambiarla secondo le tue preferenze

    # Chiama la funzione per suddividere il CSV
    split_csv(input_file=input_csv,
              output_prefix=output_pref,
              chunk_size=100000,
              output_dir=output_directory)