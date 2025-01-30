from typing import List
from neo4j import GraphDatabase
from ollama import ChatResponse
from dev.llm.ollama import Ollama
from dev.rags import RAG
import spacy
from itertools import permutations

from dev.rags.vectorrag import SYSTEM_PROMPT_VRAG
from dev.utils.entities import Document

SYSTEM_PROMPT_GRAG = """
Sei un assistente virtuale che risponde alle domande utilizzando esclusivamente il contesto fornito. 
Il contesto è costituito da dati recuperati da un grafo di conoscenza.

Regole di esecuzione:
1. Analizza attentamente il contesto fornito e identifica le informazioni più rilevanti rispetto alla domanda.
2. Considera solo le informazioni che sono strettamente correlate agli elementi chiave della query.
3. Ignora informazioni ridondanti, non pertinenti o che potrebbero confondere l'utente.
4. Genera una risposta che sia chiara, accurata e con una lunghezza approssimativa di 50-70 parole.
5. Se il contesto non fornisce informazioni sufficienti o pertinenti, dì esplicitamente "Non lo so."

Formato della risposta:
Risposta: <<Risposta generata dal modello>>
"""

class GraphRAG(RAG):
    """Implementazione di Retrieval-Augmented Generation (RAG) utilizzando un grafo di conoscenza basato su Neo4j."""

    def __init__(self, uri: str, user: str, password: str, database: str, llm_name: str, nlp_name):
        """
        Inizializza l'architettura RAG con un database Neo4j e un modello LLM.

        :param uri: URI per la connessione al database Neo4j.
        :param user: Nome utente per l'autenticazione al database Neo4j.
        :param password: Password per l'autenticazione al database Neo4j.
        :param database: Nome del database Neo4j da utilizzare.
        :param llm_name: Nome del modello LLM da utilizzare.
        :param nlp_name: Nome del modello Spacy per l'estrazione delle entità.
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.llm = Ollama(model_name=llm_name)
        self.database = database
        self.nlp = spacy.load(nlp_name)

    def compose_cypher_from_query(self, query: str) -> str:
        """
        Componi una query Cypher a partire da una domanda in linguaggio naturale.

        :param query: Query testuale fornita dall'utente.
        :return: Stringa contenente la query Cypher generata.
        """
        entities = [ent.text for ent in self.nlp(query).ents]
        base_query = """
                    MATCH (s:Entita)-[r]->(o:Entita)
                    WHERE {}
                    RETURN 
                      s.nome AS Soggetto, 
                      type(r) AS VerboOriginale, 
                      o.nome AS Oggetto
                    """

        if len(entities) == 1:
            where_clause = f"(toLower(s.nome) = \"{entities[0].lower()}\" OR toLower(o.nome) = \"{entities[0].lower()}\")"
        elif len(entities) > 1:
            # Caso con più entità: genera tutte le combinazioni di coppie (ordine considerato)
            conditions = []
            for ent1, ent2 in permutations(entities, 2):
                condition = f"(toLower(s.nome) = \"{ent1.lower()}\" AND toLower(o.nome) = \"{ent2.lower()}\")"
                conditions.append(condition)
            where_clause = " OR ".join(conditions)
        else:
            # Caso in cui non ci siano entità
            return ""
        cypher_query = base_query.format(where_clause)
        return cypher_query

    def load_data(self):
        """Metodo placeholder per il caricamento dei dati nel database Neo4j."""
        # Implementa la logica di caricamento dei dati nel database Neo4j se necessario
        NotImplemented()

    def search(self, query: str, **kwargs) -> List[Document]:
        """
        Esegue una ricerca all'interno del grafo di conoscenza utilizzando una query Cypher.

        :param query: Query testuale fornita dall'utente.
        :return: Lista di documenti contenenti i dati recuperati dal grafo.
        """
        cypher_query = self.compose_cypher_from_query(query)
        if len(cypher_query)>0:
            with self.driver.session(database=self.database) as session:
                result = session.run(cypher_query)
                return [
                    Document(
                        text=" ".join([elem['Soggetto'], elem['VerboOriginale'], elem['Oggetto']]).strip(),
                        metadata={}
                    )
                    for elem in result.data()
                ]
        else:
            return [Document(text="", metadata={"error":"Errore: Non è stata recuperata alcuna informaizone"})]

    def generate_answer(self, query: str, context: List[Document]) -> ChatResponse:
        """
       Genera una risposta basata sui dati recuperati dal grafo.

       :param query: Query testuale dell'utente.
       :param context: Lista di documenti contenenti i dati recuperati dal grafo.
       :return: Risposta generata dal modello LLM.
       """
        context_str = "\n".join([f"{doc.text}" for doc in context])
        context_gen=f"{SYSTEM_PROMPT_GRAG}\nContesto:\n{context_str}"
        return self.llm.generate(query=query, context=context_gen)


if __name__ == "__main__":
    uri = "bolt://localhost:7687"  # Modifica se necessario
    user = "neo4j"  # Modifica con il tuo username
    password = "multi_rags_1234"  # Modifica con la tua password
    database = "kgrag"  # Nome del database target
    query = "Cosa è successo in passato in Giappone?"

    graph_rag = GraphRAG(
        uri=uri, user=user, password=password, database=database,
        llm_name='llama3.1:latest', nlp_name="it_core_news_md"
    )

    retrieved = graph_rag.search(
        query=query
    )

    response_gen = graph_rag.generate_answer(query=query, context=retrieved)

    print(f"Risposta generata: {response_gen['message']['content']}")
    print(
        f"""Tempo impiegato dal modello per generare la risposta: {
        response_gen.total_duration / (10 ** 9) if response_gen.total_duration is not None else 'Non pervenuto'
        }""")
