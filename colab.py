import json
import time
from pathlib import Path

from beir.datasets.data_loader import GenericDataLoader
from whoosh import index, scoring
from whoosh.analysis import StemmingAnalyzer
from whoosh.query import Or, Term


DATASET_NAME = "fiqa"
SPLIT = "test"

BASE_DIR = Path.cwd()
DATASET_DIR = BASE_DIR / "datasets" / DATASET_NAME
INDEX_DIR = BASE_DIR / "whoosh_indices" / "fiqa_whoosh"
RESULTS_FILE = BASE_DIR / "bm25_results.json"
BM25_K1 = 1.5
BM25_B = 0.75
TITLE_BOOST = 2.0
RESULT_LIMIT = 100

ANALYZER = StemmingAnalyzer()

Queries = dict[str, str]
RankingResults = dict[str, dict[str, float]]


def load_queries(dataset_path: Path) -> Queries:
    """Carica le query dello split di test."""
    dataset = GenericDataLoader(
        data_folder=str(dataset_path)
    ).load(split=SPLIT)
    queries = dataset[1]
    return queries


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
    results: RankingResults = {}

    start_time = time.perf_counter()
    with whoosh_index.searcher(weighting=weighting) as searcher:
        total_queries = len(queries)
        for position, (query_id, text) in enumerate(queries.items(), start=1):
            query = build_query(text)
            hits = searcher.search(query, limit=RESULT_LIMIT)
            results[query_id] = {
                hit["doc_id"]: float(hit.score) for hit in hits
            }
            if position % 50 == 0 or position == total_queries:
                print(f"Ranking: {position}/{total_queries}")
    ranking_seconds = time.perf_counter() - start_time

    return results, ranking_seconds


def save_results(results: RankingResults) -> None:
    """Salva i risultati nel formato richiesto da evaluate.py."""
    with RESULTS_FILE.open("w", encoding="utf-8") as file:
        json.dump(results, file)


def main() -> None:
    queries = load_queries(DATASET_DIR)
    results, ranking_seconds = run_retrieval(queries)
    save_results(results)
    print(f"Risultati salvati in: {RESULTS_FILE}")
    print(f"Tempo ranking: {ranking_seconds:.2f} secondi")


if __name__ == "__main__":
    main()
