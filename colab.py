import time
from pathlib import Path

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from whoosh import index, scoring
from whoosh.analysis import StemmingAnalyzer
from whoosh.query import Or, Term


DATASET_NAME = "fiqa"
SPLIT = "test"

BASE_DIR = Path.cwd()
DATASET_DIR = BASE_DIR / "datasets" / DATASET_NAME
INDEX_DIR = BASE_DIR / "whoosh_indices" / "fiqa_whoosh"
BM25_K1 = 1.5
BM25_B = 0.75
TITLE_BOOST = 2.0
K_VALUES = [100]

ANALYZER = StemmingAnalyzer()

Queries = dict[str, str]
Qrels = dict[str, dict[str, int]]
RankingResults = dict[str, dict[str, float]]
Metrics = dict[str, dict[str, float] | dict[str, int | float]]


def load_dataset(dataset_path: Path) -> tuple[Queries, Qrels]:
    """Carica corpus, query e giudizi di rilevanza dello split di test."""
    dataset = GenericDataLoader(
        data_folder=str(dataset_path)
    ).load(split=SPLIT)
    queries = dataset[1]
    qrels = dataset[2]
    return queries, qrels


def build_query(text: str) -> Or:
    """Cerca ogni termine sia nel titolo sia nel contenuto."""
    terms = dict.fromkeys(token.text for token in ANALYZER(text))
    clauses = []
    for term in terms:
        clauses.append(Term("title", term, boost=TITLE_BOOST))
        clauses.append(Term("content", term))
    return Or(clauses)


def run_retrieval(queries: Queries) -> tuple[RankingResults, float]:
    """Esegue le query e misura il tempo di ranking complessivo."""
    whoosh_index = index.open_dir(INDEX_DIR)
    weighting = scoring.BM25F(K1=BM25_K1, B=BM25_B)
    result_limit = max(K_VALUES)
    results: RankingResults = {}

    start_time = time.perf_counter()
    with whoosh_index.searcher(weighting=weighting) as searcher:
        total_queries = len(queries)
        for position, (query_id, text) in enumerate(queries.items(), start=1):
            query = build_query(text)
            hits = searcher.search(query, limit=result_limit)
            results[query_id] = {
                hit["doc_id"]: float(hit.score) for hit in hits
            }
            if position % 50 == 0 or position == total_queries:
                print(f"Ranking: {position}/{total_queries}")
    ranking_seconds = time.perf_counter() - start_time

    return results, ranking_seconds


def evaluate_results(
    qrels: Qrels, results: RankingResults
) -> Metrics:
    """Calcola le metriche BEIR sui risultati prodotti."""
    evaluated_qrels = {
        query_id: qrels[query_id] for query_id in results if query_id in qrels
    }
    ndcg, mean_ap, recall, precision = EvaluateRetrieval.evaluate(
        evaluated_qrels, results, K_VALUES, ignore_identical_ids=False
    )
    return {
        "ndcg": ndcg,
        "map": mean_ap,
        "recall": recall,
        "precision": precision,
    }


def print_metrics(metrics: Metrics) -> None:
    """Stampa le metriche principali."""
    print("\nRisultati principali:")
    print(f"  nDCG@100    = {metrics['ndcg'].get('NDCG@100', 0.0):.5f}")
    print(f"  MAP@100     = {metrics['map'].get('MAP@100', 0.0):.5f}")
    print(f"  Recall@100  = {metrics['recall'].get('Recall@100', 0.0):.5f}")
    print(f"  P@100       = {metrics['precision'].get('P@100', 0.0):.5f}")
    print(
        "  Tempo ranking "
        f"= {metrics['time']['ranking_seconds']:.2f} secondi"
    )


def main() -> None:
    queries, qrels = load_dataset(DATASET_DIR)
    results, ranking_seconds = run_retrieval(queries)
    metrics = evaluate_results(qrels, results)
    metrics["time"] = {
        "ranking_seconds": ranking_seconds,
    }
    print_metrics(metrics)


if __name__ == "__main__":
    main()
