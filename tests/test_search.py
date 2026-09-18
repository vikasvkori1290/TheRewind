from rewind.search import BM25, coverage, query_terms, tokenize


def test_tokenize_keeps_identifiers_and_numbers():
    assert tokenize("def compute_late_fee(days, rate=0.75)") == [
        "def", "compute_late_fee", "days", "rate", "0.75"]
    assert tokenize("Limit is 1,237 requests.") == ["limit", "is", "1237", "requests"]


def test_query_terms_drop_stopwords_and_duplicates():
    assert query_terms("What was the exact rate limit? the rate") == ["rate", "limit"]
    assert query_terms("what is the") == []


def test_coverage():
    assert coverage(["a1", "b2"], {"a1", "x"}) == 0.5
    assert coverage([], {"a1"}) == 0.0


def test_bm25_prefers_matching_documents_and_is_positive_for_tiny_corpora():
    bm = BM25([tokenize("rate limit 1237"), tokenize("release manager priya")])
    s = bm.scores(["priya"])
    assert s[1] > 0 and s[0] == 0
    single = BM25([tokenize("only doc")])
    assert single.scores(["only"])[0] > 0
