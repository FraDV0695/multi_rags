import gc
import pandas as pd
import spacy
from spacy.matcher import DependencyMatcher
from pathlib import Path
from typing import List, Tuple
import logging

from tqdm import tqdm

# Configurazione del logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Dizionario con tutti i pattern per il DependencyMatcher
PATTERN_MATCHER = {
    "SVO_Attivo_Semplice": [
        {"RIGHT_ID": "verbo", "RIGHT_ATTRS": {"POS": "VERB", "DEP": "ROOT"}},
        {"LEFT_ID": "verbo", "REL_OP": ">", "RIGHT_ID": "soggetto", "RIGHT_ATTRS": {"DEP": "nsubj", "POS": {"IN": ["NOUN", "PROPN", "PRON"]}}},
        {"LEFT_ID": "verbo", "REL_OP": ">", "RIGHT_ID": "oggetto", "RIGHT_ATTRS": {"DEP": {"IN": ["obj", "dobj", "obl"]}, "POS": {"IN": ["NOUN", "PROPN", "PRON"]}}}
    ],
    "SVO_Attivo_Composto": [
        {"RIGHT_ID": "verbo", "RIGHT_ATTRS": {"POS": "VERB", "DEP": "ROOT"}},
        {"LEFT_ID": "verbo", "REL_OP": ">", "RIGHT_ID": "ausiliare", "RIGHT_ATTRS": {"POS": "AUX", "DEP": "aux"}},
        {"LEFT_ID": "verbo", "REL_OP": ">", "RIGHT_ID": "soggetto", "RIGHT_ATTRS": {"DEP": "nsubj", "POS": {"IN": ["NOUN", "PROPN", "PRON"]}}},
        {"LEFT_ID": "verbo", "REL_OP": ">", "RIGHT_ID": "oggetto", "RIGHT_ATTRS": {"DEP": {"IN": ["obj", "dobj", "obl"]}, "POS": {"IN": ["NOUN", "PROPN", "PRON"]}}}
    ],
    "SVO_Passivo": [
        {"RIGHT_ID": "verbo", "RIGHT_ATTRS": {"POS": "VERB", "DEP": "ROOT"}},
        {"LEFT_ID": "verbo", "REL_OP": ">", "RIGHT_ID": "ausiliare", "RIGHT_ATTRS": {"POS": "AUX", "DEP": {"IN": ["aux", "aux:pass"]}}},
        {"LEFT_ID": "verbo", "REL_OP": ">", "RIGHT_ID": "soggetto_passivo", "RIGHT_ATTRS": {"DEP": "nsubj:pass", "POS": {"IN": ["NOUN", "PROPN", "PRON"]}}},
        {"LEFT_ID": "verbo", "REL_OP": ">>", "RIGHT_ID": "agente", "RIGHT_ATTRS": {"DEP": {"IN": ["agent", "obl:agent"]}, "POS": {"IN": ["NOUN", "PROPN", "PRON"]}}}
    ],
    "Relazione_Frase_Relativa": [
        {"RIGHT_ID": "verbo", "RIGHT_ATTRS": {"POS": "VERB", "DEP": "relcl"}},
        {"LEFT_ID": "verbo", "REL_OP": ">", "RIGHT_ID": "soggetto", "RIGHT_ATTRS": {"DEP": "nsubj", "POS": {"IN": ["NOUN", "PROPN", "PRON"]}}},
        {"LEFT_ID": "verbo", "REL_OP": ">", "RIGHT_ID": "oggetto", "RIGHT_ATTRS": {"DEP": {"IN": ["obj", "dobj"]}, "POS": {"IN": ["NOUN", "PROPN"]}}}
    ],
    "Relazione_Temporale": [
        {"RIGHT_ID": "verbo", "RIGHT_ATTRS": {"POS": "VERB", "DEP": "ROOT"}},
        {"LEFT_ID": "verbo", "REL_OP": ">", "RIGHT_ID": "soggetto", "RIGHT_ATTRS": {"DEP": "nsubj", "POS": {"IN": ["NOUN", "PROPN"]}}},
        {"LEFT_ID": "verbo", "REL_OP": ">", "RIGHT_ID": "oggetto", "RIGHT_ATTRS": {"DEP": {"IN": ["obj", "obl"]}, "POS": {"IN": ["NOUN", "PROPN"]}}},
        {"LEFT_ID": "verbo", "REL_OP": ">", "RIGHT_ID": "tempo", "RIGHT_ATTRS": {"DEP": {"IN": ["obl:tmod", "nmod:tmod"]}, "POS": {"IN": ["NOUN", "NUM", "ADJ"]}}}
    ],
    "Relazione_Appartenenza": [
        {"RIGHT_ID": "possesso", "RIGHT_ATTRS": {"DEP": "nmod", "POS": "NOUN"}},
        {"LEFT_ID": "possesso", "REL_OP": ">", "RIGHT_ID": "proprietario", "RIGHT_ATTRS": {"DEP": {"IN": ["nmod:poss"]}, "POS": {"IN": ["PROPN", "NOUN"]}}}
    ],
    "Relazione_Causale": [
        {"RIGHT_ID": "verbo", "RIGHT_ATTRS": {"POS": "VERB", "DEP": "ROOT"}},
        {"LEFT_ID": "verbo", "REL_OP": ">", "RIGHT_ID": "causa", "RIGHT_ATTRS": {"DEP": "obl", "POS": "NOUN"}},
        {"LEFT_ID": "causa", "REL_OP": ">", "RIGHT_ID": "motivazione", "RIGHT_ATTRS": {"DEP": {"IN": ["case", "obl:caus"]}, "POS": {"IN": ["NOUN", "PROPN"]}}}
    ],
}

def initialize_spacy() -> Tuple[spacy.Language, DependencyMatcher]:
    """
    Inizializza il modello spaCy e configura i pattern per il DependencyMatcher.
    :return: Modello spaCy e Matcher configurato.
    """
    logging.info("Caricamento del modello spaCy...")
    nlp = spacy.load("it_core_news_md")
    matcher = DependencyMatcher(nlp.vocab)

    # Aggiungi i pattern al matcher
    for match_id, pattern in PATTERN_MATCHER.items():
        matcher.add(match_id, [pattern])
    logging.info("Matcher configurato con i pattern.")
    return nlp, matcher

def save_to_csv(data: List[Tuple], output_path: Path, columns: List[str], mode: str = "a"):
    """
    Salva i dati in un file CSV.
    :param data: Lista di tuple da salvare.
    :param output_path: Percorso del file CSV.
    :param columns: Nomi delle colonne del CSV.
    :param mode: Modalità di scrittura ("w" per sovrascrivere, "a" per appendere).
    """
    if not data:
        return
    df = pd.DataFrame(data, columns=columns)
    if not output_path.exists() or mode == "w":
        df.to_csv(output_path, index=False)
    else:
        df.to_csv(output_path, mode="a", header=False, index=False)
    logging.info(f"Dati salvati in {output_path} ({len(data)} righe).")


def build_relation(token_map: dict, pattern_name: str, url: str) -> Tuple:
    """
    Costruisce una relazione basata sul pattern specificato.

    :param token_map: Mappa tra i RIGHT_ID dei token e i token stessi.
    :param pattern_name: Nome del pattern che ha generato il match.
    :param url: URL da associare alla relazione.
    :return: Tupla che rappresenta una relazione, o None se non può essere costruita.
    """
    if pattern_name == "SVO_Attivo_Semplice":
        soggetto = token_map.get("soggetto")
        verbo = token_map.get("verbo")
        oggetto = token_map.get("oggetto")
        if soggetto and verbo and oggetto:
            tempo_verbale = verbo.morph.get("Tense")
            tempo = tempo_verbale[0] if tempo_verbale else "Sconosciuto"
            return soggetto.text.strip(), verbo.text, verbo.lemma_, tempo, oggetto.text.strip(), url
        elif pattern_name == "SVO_Attivo_Composto":
            soggetto = token_map.get("soggetto")
            verbo_principale = token_map.get("verbo_principale")
            ausiliare = token_map.get("ausiliare")
            oggetto = token_map.get("oggetto")
            if soggetto and verbo_principale and ausiliare and oggetto:
                verbo_originale = f"{ausiliare.text} {verbo_principale.text}"
                verbo_lemma = f"{ausiliare.lemma_} {verbo_principale.lemma_}"
                tempo_verbale = ausiliare.morph.get("Tense")
                tempo = tempo_verbale[0] if tempo_verbale else "Sconosciuto"
                return soggetto.text, verbo_originale, verbo_lemma, tempo, oggetto.text, url
        elif pattern_name == "SVO_Passivo":
            soggetto_passivo = token_map.get("soggetto_passivo")
            verbo_principale = token_map.get("verbo_principale")
            ausiliare = token_map.get("ausiliare")
            agente = token_map.get("agente")
            if soggetto_passivo and verbo_principale and ausiliare and agente:
                verbo_originale = f"{ausiliare.text} {verbo_principale.text}"
                verbo_lemma = f"{ausiliare.lemma_} {verbo_principale.lemma_}"
                tempo_verbale = ausiliare.morph.get("Tense")
                tempo = tempo_verbale[0] if tempo_verbale else "Sconosciuto"
                return agente.text, verbo_originale, verbo_lemma, tempo, soggetto_passivo.text, url

        elif pattern_name == "Copula_Pattern":
            soggetto = token_map.get("soggetto")
            predicato_nominale = token_map.get("predicato_nominale")
            copula = token_map.get("copula")
            if soggetto and predicato_nominale and copula:
                tempo_verbale = copula.morph.get("Tense")
                tempo = tempo_verbale[0] if tempo_verbale else "Sconosciuto"
                return  soggetto.text, copula.text, copula.lemma_, tempo, predicato_nominale.text, url

        elif pattern_name == "Relazione_Appartenenza":
            possesso = token_map.get("possesso")
            proprietario = token_map.get("proprietario")

            if possesso and proprietario:
                return  proprietario.text.strip(), "possiede", "possedere",  "Sconosciuto",  possesso.text.strip(), url



def process_chunk(chunk: pd.DataFrame, nlp: spacy.Language, matcher: DependencyMatcher) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Elabora un chunk di dati per estrarre entità e relazioni.
    :param chunk: Chunk del DataFrame.
    :param nlp: Modello spaCy.
    :param matcher: DependencyMatcher configurato.
    :return: Liste di entità e relazioni estratte.
   """


    texts = chunk['text'].tolist()
    urls = chunk['url'].tolist()

    entita_chunk = []
    relazioni_chunk = []

    # Elaborazione dei testi utilizzando nlp.pipe()
    for doc, url in zip(nlp.pipe(texts, batch_size=100, n_process=2), urls):
        # Assicurarsi che il parser sia stato applicato
        if not doc.has_annotation("DEP"):
            continue  # Salta i documenti che non possono essere analizzati

        # Estrazione delle entità
        entita_chunk.extend([(ent.text.strip(), ent.label_) for ent in doc.ents])

        # Estrazione delle relazioni
        matches = matcher(doc)
        for match_id, token_ids in matches:
            nome_pattern = doc.vocab.strings[match_id]
            # Crea una mappa tra RIGHT_ID e token
            token_map =  {PATTERN_MATCHER[nome_pattern][i]["RIGHT_ID"]: doc[token_id] for i, token_id in enumerate(token_ids)}

            if nome_pattern == "SVO_Attivo_Semplice":
                soggetto = token_map.get("soggetto")
                verbo = token_map.get("verbo")
                oggetto = token_map.get("oggetto")
                if soggetto and verbo and oggetto:
                    tempo_verbale = verbo.morph.get("Tense")
                    tempo = tempo_verbale[0] if tempo_verbale else "Sconosciuto"
                    relazioni_chunk.append((soggetto.text, verbo.text, verbo.lemma_, tempo, oggetto.text, url))

            elif nome_pattern == "SVO_Attivo_Composto":
                soggetto = token_map.get("soggetto")
                verbo_principale = token_map.get("verbo_principale")
                ausiliare = token_map.get("ausiliare")
                oggetto = token_map.get("oggetto")
                if soggetto and verbo_principale and ausiliare and oggetto:
                    verbo_originale = f"{ausiliare.text} {verbo_principale.text}"
                    verbo_lemma = f"{ausiliare.lemma_} {verbo_principale.lemma_}"
                    tempo_verbale = ausiliare.morph.get("Tense")
                    tempo = tempo_verbale[0] if tempo_verbale else "Sconosciuto"
                    relazioni_chunk.append((soggetto.text, verbo_originale,  verbo_lemma,  tempo,  oggetto.text, url))

            elif nome_pattern == "SVO_Passivo":
                soggetto_passivo = token_map.get("soggetto_passivo")
                verbo_principale = token_map.get("verbo_principale")
                ausiliare = token_map.get("ausiliare")
                agente = token_map.get("agente")
                if soggetto_passivo and verbo_principale and ausiliare and agente:
                    verbo_originale = f"{ausiliare.text} {verbo_principale.text}"
                    verbo_lemma = f"{ausiliare.lemma_} {verbo_principale.lemma_}"
                    tempo_verbale = ausiliare.morph.get("Tense")
                    tempo = tempo_verbale[0] if tempo_verbale else "Sconosciuto"
                    relazioni_chunk.append((agente.text, verbo_originale, verbo_lemma, tempo,  soggetto_passivo.text, url))

            elif nome_pattern == "Copula_Pattern":
                soggetto = token_map.get("soggetto")
                predicato_nominale = token_map.get("predicato_nominale")
                copula = token_map.get("copula")
                if soggetto and predicato_nominale and copula:
                    tempo_verbale = copula.morph.get("Tense")
                    tempo = tempo_verbale[0] if tempo_verbale else "Sconosciuto"
                    relazioni_chunk.append((soggetto.text, copula.text, copula.lemma_, tempo, predicato_nominale.text, url))

            elif nome_pattern == "Relazione_Frase_Relativa":
                soggetto = token_map.get("soggetto")
                verbo = token_map.get("verbo")
                oggetto = token_map.get("oggetto")
                if soggetto and verbo and oggetto:
                    relazioni_chunk.append((soggetto.text.strip(), verbo.text.strip(), verbo.lemma_, "Relativo",
                                            oggetto.text.strip(), url))

            elif nome_pattern == "Relazione_Temporale":
                soggetto = token_map.get("soggetto")
                verbo = token_map.get("verbo")
                oggetto = token_map.get("oggetto")
                tempo = token_map.get("tempo")
                if soggetto and verbo and oggetto and tempo:
                    relazioni_chunk.append((soggetto.text.strip(), verbo.text.strip(), verbo.lemma_, tempo.text.strip(),
                                            oggetto.text.strip(), url))

            elif nome_pattern == "Relazione_Appartenenza":
                possesso = token_map.get("possesso")
                proprietario = token_map.get("proprietario")
                if possesso and proprietario:
                    relazioni_chunk.append(
                        (proprietario.text.strip(), "possiede", possesso.text.strip(), "Appartenenza", "", url))

            elif nome_pattern == "Relazione_Causale":
                verbo = token_map.get("verbo")
                causa = token_map.get("causa")
                motivazione = token_map.get("motivazione")
                if verbo and causa and motivazione:
                    relazioni_chunk.append(
                        (causa.text.strip(), verbo.text.strip(), verbo.lemma_, motivazione.text.strip(), "", url))

    # Deduplicazione delle entità
    entita_chunk = list(set(entita_chunk))

    return entita_chunk, relazioni_chunk


def process_dataset(file_path: str, output_dir: str, chunk_size: int = 5000):
    """
    Elabora un dataset CSV per estrarre entità e relazioni in blocchi.
    :param file_path: Percorso al file CSV di input.
    :param output_dir: Directory per i file CSV di output.
    :param chunk_size: Numero di righe per chunk.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    entita_file = output_path / "entita.csv"
    relazioni_file = output_path / "relazioni.csv"

    if entita_file.exists():
        entita_file.unlink()
    if relazioni_file.exists():
        relazioni_file.unlink()

    nlp, matcher = initialize_spacy()

    logging.info("Inizio elaborazione del dataset...")
    reader = pd.read_csv(file_path, chunksize=chunk_size)

    for chunk_number, chunk in enumerate(tqdm(reader, desc="Processando in chunk..."), start=1):
        logging.info(f"Elaborazione del chunk {chunk_number} con {len(chunk)} righe...")
        if "id" not in chunk.columns or "text" not in chunk.columns:
            logging.warning(f"Chunk {chunk_number} privo delle colonne 'id' o 'text'. Saltando.")
            continue

        entita_chunk, relazioni_chunk = process_chunk(chunk, nlp, matcher)

        save_to_csv(entita_chunk, entita_file, columns=["entita", "tipo"])
        save_to_csv(relazioni_chunk, relazioni_file, columns=["soggetto", "verbo_originale", "verbo_lemma", "tempo", "oggetto", "url"])

        gc.collect()


if __name__ == "__main__":
    file_path = '../../data/subset_dataset_gcp_cleaned.csv'  # Percorso al dataset
    output_dir = '../../data/output_neo4j'  # Directory per i file CSV
    process_dataset(file_path, output_dir, chunk_size=1000)

    entita = pd.read_csv(f'{output_dir}/entita.csv')
    relazioni = pd.read_csv(f'{output_dir}/relazioni.csv')

    caratteri_speciali = r'[^a-zA-Z0-9]'
    emoji_pattern = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]'

    maschera = entita['entita'].fillna('').astype(str).str.contains(
        rf'^{caratteri_speciali}|{caratteri_speciali}$|{emoji_pattern}', regex=True
    )
    entita = entita[~maschera].reset_index(drop=True)

    entita_pivot = entita.pivot_table(index='entita', columns='tipo', aggfunc='size', fill_value=0)

    tipo_piu_frequente = entita_pivot.idxmax(axis=1).reset_index()
    tipo_piu_frequente.columns = ['entita', 'tipo_piu_frequente']

    entita_filtrato = entita.merge(tipo_piu_frequente, on='entita')
    entita_filtrato = entita_filtrato[entita_filtrato['tipo'] == entita_filtrato['tipo_piu_frequente']].drop(
        columns=['tipo_piu_frequente'])

    entita_filtrato = entita_filtrato.drop_duplicates(subset='entita').reset_index(drop=True)

    relazioni_filtrato = relazioni[
        ~relazioni['tempo'].fillna('').str.contains(r'Attributiva|Nominale|Luogo|Copula', regex=True)].drop_duplicates(
        relazioni.columns)

    for col in relazioni_filtrato.columns:
        relazioni_filtrato = relazioni_filtrato[~relazioni_filtrato[col].str.contains(';', na=False)]

    relazioni_filtrato = relazioni_filtrato.map(lambda x: str(x).replace('"', '') if isinstance(x, str) else x)

    entita_relazioni = set(relazioni_filtrato['soggetto']).union(set(relazioni_filtrato['oggetto']))

    entita_filerby_relazioni = entita_filtrato[entita_filtrato['entita'].isin(entita_relazioni)].reset_index(drop=True)

    entita_filerby_relazioni.to_csv(f"{output_dir}/entita_filtrate.csv", index=False)
    relazioni_filtrato.to_csv(f"{output_dir}/relazioni_filtrate.csv", index=False)