import time
from typing import Dict, Any, List

import numpy as np
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
from sklearn.metrics import precision_score, recall_score, accuracy_score
from tqdm import tqdm

from dev.rags import RAG

import pandas as pd
from dev.rags.vectorrag import VectorRAG
from dev.rags.vectorrag_reranking import VectorRAGReranking
from dev.rags.graphrag import GraphRAG
from dev.utils.entities import Document


class EvaluationMetrics:
    """
    Classe per l'analisi delle metriche di valutazione per architetture RAG.
    """

    def __init__(self, rag_system: RAG):
        """
        Inizializza l'oggetto RAGEvaluator con un sistema RAG specifico.
        """
        self.rag_system = rag_system
        self.rouge_evaluator = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # -------------------------
    # Metriche di Recupero
    # -------------------------
    def evaluate_retrieval(self, queries:List[str], ground_truth_docs:[], k=5) -> Dict[str, Any]:
        """
        Valuta le metriche di recupero per il sistema RAG.
        """
        precisions, recalls, accuracies, reciprocal_ranks, average_precisions = [], [], [], [], []

        for idx, query in enumerate(queries):
            retrieved_docs = self.rag_system.search(query, top_k=k)
            retrieved_doc_ids = [doc['id'] for doc in retrieved_docs]
            relevant_doc_ids = ground_truth_docs[idx]

            y_true = [1 if doc_id in relevant_doc_ids else 0 for doc_id in retrieved_doc_ids]
            y_pred = [1] * len(retrieved_doc_ids)

            accuracies.append(accuracy_score(y_true, y_pred))
            precisions.append(precision_score(y_true, y_pred, zero_division=0))
            recalls.append(recall_score(y_true, y_pred, zero_division=0))
            reciprocal_ranks.append(self.reciprocal_rank(retrieved_doc_ids, relevant_doc_ids))
            average_precisions.append(self.average_precision(retrieved_doc_ids, relevant_doc_ids))

        return {
            'Accuracy': np.mean(accuracies),
            'Precision@k': np.mean(precisions),
            'Recall@k': np.mean(recalls),
            'MRR': np.mean(reciprocal_ranks),
            'MAP': np.mean(average_precisions)
        }

    def reciprocal_rank(self, retrieved_doc_ids, relevant_doc_ids):
        for idx, doc_id in enumerate(retrieved_doc_ids):
            if doc_id in relevant_doc_ids:
                return 1 / (idx + 1)
        return 0

    def average_precision(self, retrieved_doc_ids, relevant_doc_ids):
        num_relevant, score = 0, 0.0
        for idx, doc_id in enumerate(retrieved_doc_ids):
            if doc_id in relevant_doc_ids:
                num_relevant += 1
                score += num_relevant / (idx + 1)
        return score / num_relevant if num_relevant > 0 else 0

    # ----------------------------------
    # Metriche di Generazione e Latenza
    # -----------------------------------
    def evaluate_generation(self, queries: List[str], ground_truth_answers: List[str], k=5) -> Dict[str, Any]:
        """
        Valuta le metriche di generazione per il sistema RAG.
        """
        bleu_scores, rouge_scores, bert_scores = [], [], []
        latencies = []

        for idx, query in enumerate(tqdm(queries, desc="Eseguo query...")):
            start_time = time.time()
            retrieved_docs: List[Document] = self.rag_system.search(query, top_k=k)
            generated_answer = self.rag_system.generate_answer(query, retrieved_docs)
            latencies.append(time.time() - start_time)

            reference = ground_truth_answers[idx]
            candidate = generated_answer['message']['content']

            bleu_scores.append(
                sentence_bleu(
                    references=[reference.split()],
                    hypothesis=candidate.split(),
                    smoothing_function=SmoothingFunction().method1
                )
            )

            rouge_score = self.rouge_evaluator.score(reference, candidate)
            rouge_scores.append(rouge_score['rougeL'].fmeasure)

            P, R, F1 = bert_score([candidate], [reference], lang='it')
            bert_scores.append(F1.mean().item())

        return {
            'BLEU': np.mean(bleu_scores),
            'ROUGE-L': np.mean(rouge_scores),
            'BERTScore': np.mean(bert_scores),
            'Latency': {
                'Average Latency': np.mean(latencies),
                'Max Latency': np.max(latencies),
                'Min Latency': np.min(latencies)
            }
        }

    # -------------------------
    # Report Completo
    # -------------------------
    def generate_report(self, queries, ground_truth_answers, k=5): #ground_truth_docs
        """
        Genera un report completo delle metriche di valutazione.
        """
        #retrieval_metrics = self.evaluate_retrieval(queries, ground_truth_docs, k)
        generation_metrics = self.evaluate_generation(queries, ground_truth_answers, k)

        report = {
            #'Retrieval Metrics': retrieval_metrics,
            'Generation Metrics': generation_metrics,
        }
        return report


if __name__=="__main__":
    # vrag = VectorRAG(
    #     index_path="../../database/faiss_index/documents.index",
    #     metadata_path="../../database/faiss_index/documents_metadata.parquet",
    #     llm_name="llama3.1:latest",
    #     embedding_name="sentence-transformers/all-MiniLM-L6-v2"
    # )
    vrag_rk = VectorRAGReranking(
        index_path="../../database/faiss_index/documents.index",
        metadata_path="../../database/faiss_index/documents_metadata.parquet",
        llm_name="llama3.1:latest",
        embedding_name="sentence-transformers/all-MiniLM-L6-v2",
        cross_encoder_model="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    )
    # graph_rag = GraphRAG(
    #     uri="bolt://localhost:7687", user="neo4j", password="multi_rags_1234", database="kgrag",
    #     llm_name='llama3.1:latest', nlp_name="it_core_news_md"
    # )

    # evaluator = EvaluationMetrics(rag_system=vrag)
    evaluator = EvaluationMetrics(rag_system=vrag_rk)
    # evaluator = EvaluationMetrics(rag_system=graph_rag)
    df_test = pd.read_csv(
        '../../data/test_dataset_qa.csv',
        usecols=['question', 'ground_truth_answer', 'vector_relevant_docs']
    )

    queries = df_test['question'].tolist()[:250]
    ground_truth_answers = df_test['ground_truth_answer'].tolist()[:250]

    report = evaluator.generate_report(queries, ground_truth_answers)
    flattened_data = []
    #retrieval = report['Retrieval Metrics']
    generation = report['Generation Metrics']
    flattened_data.append({
        #'Accuracy': retrieval['Accuracy'],
        #'Precision@k': retrieval['Precision@k'],
        #'Recall@k': retrieval['Recall@k'],
        #'MRR': retrieval['MRR'],
        #'MAP': retrieval['MAP'],
        'BLEU': generation['BLEU'],
        'ROUGE-L': generation['ROUGE-L'],
        'BERTScore': generation['BERTScore'],
        'Average Latency': generation['Latency']['Average Latency'],
        'Max Latency': generation['Latency']['Max Latency'],
        'Min Latency': generation['Latency']['Min Latency']
    })
    df = pd.DataFrame(flattened_data)
    df.to_csv('./results/vectorrag_rk_metrics_full.csv', index=False)
    # df.to_csv('./results/vectorrag_metrics_full.csv', index=False)
    # df.to_csv('./results/graphrag_metrics_full.csv', index=False)
    print(report)

