"""A saved self-supervised encoder can be reloaded and gives identical outputs."""
import importlib.util
from pathlib import Path

import torch


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_robust_ssl.py"
    spec = importlib.util.spec_from_file_location("run_robust_ssl_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_encoder_checkpoint_roundtrip(tmp_path):
    mod = _load_script()
    torch.manual_seed(0)
    enc = mod.Encoder().eval()
    f = tmp_path / "enc.pt"
    torch.save({"encoder": enc.state_dict()}, f)

    torch.manual_seed(1)  # different init, so equality below can only come from the loaded weights
    enc2 = mod.Encoder().eval()
    enc2.load_state_dict(torch.load(f, map_location="cpu")["encoder"])

    x = torch.randn(4, 104, 240)
    with torch.no_grad():
        assert torch.allclose(enc(x), enc2(x), atol=1e-6)