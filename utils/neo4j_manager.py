import time

from neo4j import GraphDatabase
import os
import sys


class Neo4jCSVLoader:
    def __init__(self, uri, user, password, database, csv_dir, file_prefix, file_suffix, total_files, batch_size=1000,
                 parallel=False):
        """
        Inizializza il loader Neo4j.

        :param uri: URI di connessione a Neo4j (es. bolt://localhost:7687)
        :param user: Nome utente Neo4j
        :param password: Password Neo4j
        :param database: Nome del database Neo4j (es. 'kgrag')
        :param csv_dir: Directory contenente i file CSV
        :param file_prefix: Prefisso dei file CSV (es. 'relazioni_filtrate_part')
        :param file_suffix: Suffisso dei file CSV (es. '.csv')
        :param total_files: Numero totale di file CSV da processare
        :param batch_size: Numero di righe per batch
        :param parallel: Booleano per abilitare la parallelizzazione
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.csv_dir = csv_dir
        self.file_prefix = file_prefix
        self.file_suffix = file_suffix
        self.total_files = total_files
        self.batch_size = batch_size
        self.parallel = parallel

    def close(self):
        """Chiude la connessione al database Neo4j."""
        self.driver.close()

    def create_indexes_constraints(self, session):
        """Crea indici e vincoli necessari."""
        create_index_query = """
        CREATE INDEX IF NOT EXISTS entita_nome_index FOR (n:Entita) ON (n.nome);
        """
        create_constraint_query = """
        CREATE CONSTRAINT IF NOT EXISTS entita_nome_unique FOR (n:Entita) REQUIRE n.nome IS UNIQUE;
        """
        session.run(create_index_query)
        session.run(create_constraint_query)
        print("Indici e vincoli creati o già esistenti.")

    def load_relazioni_from_file(self, session, file_name):
        """
        Esegue la query APOC per caricare le relazioni da un singolo file CSV.

        :param session: Sessione Neo4j
        :param file_name: Nome del file CSV
        """
        start_time = time.time()
        query = f"""
        CALL apoc.periodic.iterate(
          "RETURN '{file_name}' AS file",
          "
            LOAD CSV WITH HEADERS FROM 'file:///' + file AS row
            MATCH (s:Entita {{nome: row.soggetto}})
            MATCH (o:Entita {{nome: row.oggetto}})
            CALL apoc.merge.relationship(
              s, 
              row.verbo_originale, 
              {{
                tempo: row.tempo, 
                url: row.url, 
                verbo_lemma: row.verbo_lemma
              }}, 
              {{}}, 
              o
            ) YIELD rel
            RETURN rel
          ",
          {{
            batchSize: {self.batch_size}, 
            parallel: {str(self.parallel).lower()}
          }}
        )
        YIELD batches, total
        RETURN batches, total;
        """
        try:
            result = session.run(query)
            record = result.single()
            if record:
                batches = record["batches"]
                total = record["total"]
                print(f"File '{file_name}': Batch elaborati = {batches}, Relazioni create = {total}, Tempo impiegato = {time.time()-start_time} sec.")
            else:
                print(f"File '{file_name}': Nessun batch elaborato.")
        except Exception as e:
            print(f"Errore durante il caricamento del file '{file_name}': {e}")

    def load_all_files(self):
        """Carica tutte le relazioni dai file CSV."""
        with self.driver.session(database=self.database) as session:
            # Crea indici e vincoli
            # self.create_indexes_constraints(session)

            # Itera su tutti i file
            for num in range(1, self.total_files + 1):
                file_name = f"{self.file_prefix}{num}{self.file_suffix}"
                file_path = os.path.join(self.csv_dir, file_name)

                # Verifica se il file esiste
                if not os.path.isfile(file_path):
                    print(f"File '{file_name}' non trovato nella directory '{self.csv_dir}'. Salto...")
                    continue

                print(f"Inizio caricamento del file '{file_name}'...")
                self.load_relazioni_from_file(session, file_name)
                print(f"Fine caricamento del file '{file_name}'.\n")


if __name__ == "__main__":
    # Configurazione di connessione a Neo4j
    uri = "bolt://localhost:7687"  # Modifica se necessario
    user = "neo4j"  # Modifica con il tuo username
    password = "multi_rags_1234"  # Modifica con la tua password
    database = "kgrag"  # Nome del database target

    # Configurazione dei file CSV
    csv_dir = "C:\\Users\\franc\\.Neo4jDesktop\\relate-data\\dbmss\\dbms-797f8169-2207-4cc6-9fb5-9d39d87d5e71\\import"  # Modifica con il percorso corretto della directory 'import'
    file_prefix = "relazioni_filtrate_part"
    file_suffix = ".csv"
    total_files = 77  # Numero totale di file CSV

    # Configurazione di caricamento
    batch_size = 1000
    parallel = False

    # Verifica se la directory dei file CSV esiste
    if not os.path.isdir(csv_dir):
        print(f"Directory '{csv_dir}' non trovata. Assicurati che il percorso sia corretto.")
        sys.exit(1)

    # Inizializza il loader
    loader = Neo4jCSVLoader(
        uri=uri,
        user=user,
        password=password,
        database=database,
        csv_dir=csv_dir,
        file_prefix=file_prefix,
        file_suffix=file_suffix,
        total_files=total_files,
        batch_size=batch_size,
        parallel=parallel
    )

    try:
        # Esegui il caricamento di tutti i file
        loader.load_all_files()
    finally:
        # Chiudi la connessione al database
        loader.close()
