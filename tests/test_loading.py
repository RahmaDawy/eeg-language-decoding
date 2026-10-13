from eegdecode.data.inspect_data import EEG_RE, inventory


def test_name_pattern_parses_fields():
    m = EEG_RE.match("gip_ZAB_NR3_EEG.mat")
    assert m["prefix"] == "gip" and m["subject"] == "ZAB" and m["block"] == "3"


def test_name_pattern_rejects_other_files():
    assert EEG_RE.match("ZAB_NR3_corrected_ET.mat") is None
    assert EEG_RE.match("wordbounds_NR_ZAB.mat") is None


def test_inventory_finds_files_and_flags_missing(tmp_path):
    d = tmp_path / "task2 - NR" / "Preprocessed" / "ZAB"
    d.mkdir(parents=True)
    for name in ["gip_ZAB_NR1_EEG.mat", "oip_ZAB_NR2_EEG.mat"]:
        (d / name).write_bytes(b"x")
    df = inventory(tmp_path)
    assert len(df) == 2
    assert set(df.block) == {1, 2}
    assert set(df.prefix) == {"gip", "oip"}