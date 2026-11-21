from app.services.japanese import devoice, fold_sokuon, lesson_units, morae

VOCAB = {
    p: i
    for i, p in enumerate(
        "<pad> a i u e o k ɡ s z t d n m h x ɸ b p ɾ kʲ j w ʃ dʒ tʃ ts ŋ".split()
    )
}


def test_morae_split_yoon_and_long_vowels():
    assert morae("リョクチャ") == [("リョ", ["ɾ", "j", "o"]), ("ク", ["k", "u"]), ("チャ", ["tʃ", "a"])]
    assert morae("コーヒー") == [("コー", ["k", "o"]), ("ヒー", ["x", "i"])]
    assert morae("トーキョー") == [("トー", ["t", "o"]), ("キョー", ["kʲ", "o"])]


def test_sokuon_folds_into_the_next_mora():
    assert fold_sokuon(morae("ガッコー")) == [("ガ", ["ɡ", "a"]), ("ッコー", ["k", "o"])]
    assert fold_sokuon(morae("キッテ"))[1] == ("ッテ", ["t", "e"])
    assert fold_sokuon(morae("アッ")) == [("ア", ["a"])]


def test_moraic_nasal_assimilates():
    assert morae("サンポ")[1] == ("ン", ["m"])
    assert morae("サンカ")[1] == ("ン", ["ŋ"])
    assert morae("ホンヤ")[1] == ("ン", ["n"])
    assert morae("オカーサン")[-1] == ("ン", ["n"])


def test_small_vowels_and_hiragana():
    assert morae("ティー") == [("ティー", ["t", "i"])]
    assert morae("ふぁん") == [("ファ", ["ɸ", "a"]), ("ン", ["n"])]
    assert morae("ウィ") == [("ウィ", ["w", "i"])]


def test_devoicing_between_voiceless_and_at_the_end():
    assert devoice([["k", "u"], ["tʃ", "a"]]) == [["k"], ["tʃ", "a"]]
    assert devoice([["m", "a"], ["s", "u"]]) == [["m", "a"], ["s"]]
    assert devoice([["k", "u"], ["d", "a"]]) == [["k", "u"], ["d", "a"]]
    assert devoice([["k", "a"], ["t", "a"]]) == [["k", "a"], ["t", "a"]]


def test_units_per_word_from_reading():
    reading = [("私", "ワタシ"), ("は", "ワ"), ("緑茶", "リョクチャ"), ("。", "")]
    units = lesson_units("私は緑茶。", VOCAB, reading=lambda _: reading)
    assert [(u.word, u.word_index, u.label) for u in units] == [
        ("私", 0, "ワ"), ("私", 0, "タ"), ("私", 0, "シ"), ("は", 1, "ワ"),
        ("緑茶", 2, "リョ"), ("緑茶", 2, "ク"), ("緑茶", 2, "チャ"),
    ]
    assert units[4].token_ids == [VOCAB["ɾ"], VOCAB["j"], VOCAB["o"]]
    assert units[5].token_ids == [VOCAB["k"]]  # devoiced before チャ
    assert all(u.tone is None for u in units)


def test_word_without_reading_gives_up():
    reading = [("ABC", None), ("は", "ワ")]
    assert lesson_units("ABCは", VOCAB, reading=lambda _: reading) == []


def test_missing_fugashi_gives_up():
    def reading(_):
        raise ImportError("fugashi")

    assert lesson_units("緑茶", VOCAB, reading=reading) == []


def test_repeated_word_keeps_separate_positions():
    reading = [("はい", "ハイ"), ("はい", "ハイ")]
    units = lesson_units("はいはい", VOCAB, reading=lambda _: reading)
    assert [u.word_index for u in units] == [0, 0, 1, 1]
