import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# Download risorse della libreria nltk (direi serva per utilizzare stopwords e stemmer) 
nltk.download("book")


# ---------------------------------------------------------------------------
# Preprocessing NLTK
# ---------------------------------------------------------------------------

ENGLISH_STOPWORDS = set(stopwords.words("english"))
PORTER_STEMMER = PorterStemmer()


def nltk_tokenize(text: str) -> list[str]:
    # Conversione in lowercase e tokenizzazione.
    tokens = nltk.word_tokenize(text.lower())

    # Eliminazione della punteggiatura e delle stopword.
    tokens = [
        token
        for token in tokens
        if token.isalnum() and token not in ENGLISH_STOPWORDS
    ]

    # Stemming dei token rimanenti.
    tokens = [
        PORTER_STEMMER.stem(token)
        for token in tokens
    ]

    return tokens