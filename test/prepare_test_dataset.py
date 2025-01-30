import json
from tqdm import tqdm

from dev.llm.ollama import Ollama
from dev.utils.data_manager import DataCSVManager
import pandas as pd


if __name__ == "__main__":

    llm = Ollama(model_name='llama3.1:latest') #llama3.1:8b-instruct-q3_K_S
    data_manager = DataCSVManager(ref_dir="../../data")

    documents = data_manager.load_data(data_name="subset_dataset_gcp_cleaned.csv", text_column='text')

    # Prompt del sistema
    system_prompt = """Quando ricevi un testo, il tuo compito è generare due domande semplici e inerenti al contenuto del testo, insieme alle loro risposte. 
    Entrambe le domande e risposte devono seguire queste regole:

            1. Le domande devono essere semplici, dirette e basate esclusivamente sul contenuto del testo.
            2. Ogni risposta deve essere compresa tra 10 e 70 parole.
            3. Evita riferimenti espliciti al testo, come "cosa viene detto nel testo?".
            4. Fornisci il risultato nel seguente formato JSON senza errori di formattazione:            
            {{
              "qa_pairs": [
                {{
                  "question": "Prima domanda?",
                  "answer": "Risposta alla prima domanda."
                }},
                {{
                  "question": "Seconda domanda?",
                  "answer": "Risposta alla seconda domanda."
                }}
              ]
            }}
            4. Escapa correttamente le virgolette all'interno di domande e risposte.
            5. Dopo il valore della chiave "answer" non deve mai essere presente una virgola, per garantire la compatibilità con il formato JSON.
            6. Restituisci solo il risultato formattato in JSon e nient'altro
    """

    qa_data = []
    test_dataset = []
    for doc in tqdm(documents, desc='Generando QA per il dataset di test'):
        # Messaggio dell'utente con le istruzioni incluse
        user_message = f"""Dato il seguente testo:
                            {doc.text}
                        Genera due domande e risposte rispettando le regole e il formato JSON specificato.
            """

        response = llm.generate(query=user_message, context=system_prompt)

        # Estrazione della risposta dell'assistente
        assistant_reply = response['message']['content'].strip()

        # Parsing dell'output JSON
        try:
            qa_pairs=json.loads(assistant_reply)["qa_pairs"]
            for pair in qa_pairs:
                test_dataset.append({
                    "question": pair["question"],
                    "ground_truth_answer": pair["answer"],
                })
        except json.JSONDecodeError:
            print(f"Errore nel parsing del JSON per il documento {doc.metadata['id']}. Risposta dell'assistente:")
            print(assistant_reply)

    if len(test_dataset)>0:
        df = pd.DataFrame(test_dataset)
        df.to_csv("../../data/test_dataset_qa.csv", encoding="utf-8", index=False)
        print(f"File CSV salvato ")
    else:
        print("non sono state trovate domande generate dal modello.")