import importlib.util
from pathlib import Path

import torch


def _load_run_final():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_final.py"
    spec = importlib.util.spec_from_file_location("run_final", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_encoder_checkpoint_roundtrip(tmp_path):
    rf = _load_run_final()
    torch.manual_seed(0)
    enc = rf.Encoder().eval()
    x = torch.randn(4, 105, 150)
    path = tmp_path / "enc.pt"
    torch.save(enc.state_dict(), path)
    enc2 = rf.Encoder().eval()
    enc2.load_state_dict(torch.load(path))
    with torch.no_grad():
        assert torch.allclose(enc(x), enc2(x), atol=1e-6)