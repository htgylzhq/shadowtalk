from app.services.japanese import japanese_units, morae

VOCAB = {
    p: i
    for i, p in enumerate(
        "<pad> a i u e o aː iː uː eː oː k ɡ s z t d n m h ç ɸ b p ɾ rʲ kʲ j w ɕ dʑ tɕ ts ŋ".split()
    )
}


def test_morae_split_yoon_and_long_vowels():
    assert morae("リョクチャ") == [("リョ", ["rʲ", "o"]), ("ク", ["k", "u"]), ("チャ", ["tɕ", "a"])]
    assert morae("コーヒー") == [("コー", ["k", "oː"]), ("ヒー", ["ç", "iː"])]
    assert morae("トーキョー") == [("トー", ["t", "oː"]), ("キョー", ["kʲ", "oː"])]


def test_sokuon_takes_the_next_consonant():
    assert morae("ガッコー") == [("ガ", ["ɡ", "a"]), ("ッ", ["k"]), ("コー", ["k", "oː"])]
    assert morae("キッテ")[1] == ("ッ", ["t"])


def test_moraic_nasal_assimilates():
    assert morae("サンポ")[1] == ("ン", ["m"])
    assert morae("サンカ")[1] == ("ン", ["ŋ"])
    assert morae("ホンヤ")[1] == ("ン", ["n"])
    assert morae("オカーサン")[-1] == ("ン", ["n"])


def test_small_vowels_and_hiragana():
    assert morae("ティー") == [("ティー", ["t", "iː"])]
    assert morae("ふぁん") == [("ファ", ["ɸ", "a"]), ("ン", ["n"])]
    assert morae("ウィ") == [("ウィ", ["w", "i"])]


def test_units_per_word_from_reading():
    reading = [("私", "ワタシ"), ("は", "ワ"), ("緑茶", "リョクチャ"), ("。", "")]
    units = japanese_units("私は緑茶。", VOCAB, reading=lambda _: reading)
    assert [w for w, _ in units] == ["私", "は", "緑茶"]
    assert units[2][1] == [
        ("リョ", [VOCAB["rʲ"], VOCAB["o"]]),
        ("ク", [VOCAB["k"], VOCAB["u"]]),
        ("チャ", [VOCAB["tɕ"], VOCAB["a"]]),
    ]


def test_word_without_reading_gives_up():
    reading = [("ABC", None), ("は", "ワ")]
    assert japanese_units("ABCは", VOCAB, reading=lambda _: reading) == []


def test_missing_fugashi_gives_up():
    def reading(_):
        raise ImportError("fugashi")

    assert japanese_units("緑茶", VOCAB, reading=reading) == []
