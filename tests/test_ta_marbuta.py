from egypocket.text.ta_marbuta import build_lexicon, is_candidate, restore, vote


def test_candidates():
    assert is_candidate("حاجه")
    assert is_candidate("المدينه")
    assert not is_candidate("ده")  # too short
    assert not is_candidate("معاه")  # alef before ه hides the evidence
    assert not is_candidate("الله")
    assert not is_candidate("والله")
    assert not is_candidate("اتجه")
    assert not is_candidate("وبيواجه")
    assert not is_candidate("حاجة")


def test_votes_from_catt_marks():
    assert vote("حَاجَه") == 1
    assert vote("الْمَدِينَه") == 1
    assert vote("شْوَيَّه") == 1  # shadda + fatha
    assert vote("كِتَابُه") == -1
    assert vote("صَاْحْبُه") == -1
    assert vote("فِيه") == 0
    assert vote("مُنَبِّه") == 0


def test_lexicon_majority():
    pairs = [
        ("حاجه كتابه", "حَاجَه كِتَابُه"),
        ("حاجه", "حَاجَه"),
        ("حاجه", "حَاجُه"),  # one noisy vote
        ("كتابه", "كِتَابُه"),
    ]
    lexicon, votes = build_lexicon(pairs)
    assert lexicon == {"حاجه"}
    assert votes.positive["حاجه"] == 2 and votes.negative["حاجه"] == 1
    assert restore("حاجه كتابه حاجه", lexicon) == "حاجة كتابه حاجة"


def test_mismatched_pairs_ignored():
    lexicon, _ = build_lexicon([("حاجه", None), ("حاجه كبيره", "حَاجَه")])
    assert lexicon == set()
