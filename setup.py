from pathlib import Path

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from whoosh import index
from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import ID, TEXT, Schema


DATASET_NAME = "fiqa"
SPLIT = "test"

BASE_DIR = Path.cwd()
DATASETS_DIR = BASE_DIR / "datasets"
INDEX_DIR = BASE_DIR / "whoosh_indices" / "fiqa_whoosh"

ANALYZER = StemmingAnalyzer()

Corpus = dict[str, dict[str, str]]


def download_dataset() -> Path:
    """Scarica FiQA nel formato BEIR."""
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    url = (
        "https://public.ukp.informatik.tu-darmstadt.de/"
        f"thakur/BEIR/datasets/{DATASET_NAME}.zip"
    )
    print(f"Download dataset da: {url}")
    return Path(util.download_and_unzip(url, str(DATASETS_DIR)))


def load_corpus(dataset_path: Path) -> Corpus:
    """Carica il corpus necessario per costruire l'indice."""
    dataset = GenericDataLoader(
        data_folder=str(dataset_path)
    ).load(split=SPLIT)
    corpus = dataset[0]
    return corpus


def build_index(corpus: Corpus) -> None:
    """Ricrea l'indice Whoosh con titolo e contenuto dei documenti."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    schema = Schema(
        doc_id=ID(stored=True, unique=True),
        title=TEXT(analyzer=ANALYZER),
        content=TEXT(analyzer=ANALYZER),
    )

    whoosh_index = index.create_in(INDEX_DIR, schema)
    writer = whoosh_index.writer()

    total_documents = len(corpus)
    for position, (document_id, document) in enumerate(corpus.items(), start=1):
        writer.add_document(
            doc_id=document_id,
            title=document.get("title", ""),
            content=document.get("text", ""),
        )
        if position % 10000 == 0 or position == total_documents:
            print(f"Indicizzazione: {position}/{total_documents}")

    writer.commit()


def main() -> None:
    dataset_path = download_dataset()
    corpus = load_corpus(dataset_path)
    build_index(corpus)
    print(f"Setup completato. Indice creato in: {INDEX_DIR}")


if __name__ == "__main__":
    main()
