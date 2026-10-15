import numpy as np

from eegdecode.data.probe_eeg import (
    _decode, flat_str, is_missing, load_zuco2_sentences, norm_text, parse_nr_line, sentence_windows,
)


def test_decode_char_array():
    assert _decode(np.array([[72], [105]], dtype=np.uint16)) == "Hi"


def test_decode_matlab_empty_marker():
    assert _decode(np.array([0, 0], dtype=np.uint64)) is None


def test_decode_scalar():
    assert _decode(np.array([[500.0]])) == 500.0


def test_is_missing_and_flat_str():
    assert is_missing("(missing)") and not is_missing(np.array([1, 2]))
    assert flat_str(np.array([[3.0], [5.0]])) == "3,5"
    assert flat_str("(missing)") is None


def test_norm_text_ignores_case_spacing_punctuation():
    assert norm_text("Henry Ford, with  investors,formed") == norm_text("henry ford with investors, formed")


def test_parse_line_with_semicolon_in_text_and_control():
    r = parse_nr_line("2;1;He said a; then left.;CONTROL")
    assert (r["par"], r["sent"]) == (2, 1)
    assert r["text"] == "He said a; then left." and r["control"]


def test_load_zuco2_skips_control_question_files(tmp_path):
    (tmp_path / "nr_1.csv").write_text("1;1;Some text here.;\n2;1;Other text.;CONTROL\n")
    (tmp_path / "nr_1_control_questions.csv").write_text("garbage line\n")
    rows = load_zuco2_sentences(tmp_path)
    assert len(rows) == 2 and rows[0]["block"] == 1
    assert rows[1]["control"] and not rows[0]["control"]


def test_sentence_windows_pairs_in_order():
    ev = [("90", 12.0, ""), ("10", 100.0, ""), ("11", 400.0, ""), ("10", 404.0, ""), ("11", 900.0, "")]
    wins, problems = sentence_windows(ev)
    assert wins == [(100.0, 400.0), (404.0, 900.0)] and problems == 0


def test_sentence_windows_flags_unpaired():
    ev = [("10", 1.0, ""), ("10", 5.0, ""), ("11", 9.0, ""), ("11", 12.0, "")]
    wins, problems = sentence_windows(ev)
    assert len(wins) == 1 and problems == 2