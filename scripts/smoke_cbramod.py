"""Smoke test: can the pretrained CBraMod load, and what does it output for a (batch, 104 channels, 400 samples) input?"""
import torch
from braindecode.models import CBraMod

x = torch.randn(2, 104, 400)  # 2 s at 200 Hz
tries = [
    dict(n_chans=104, n_times=400, sfreq=200, return_encoder_output=True),
    dict(n_chans=104, input_window_seconds=2, sfreq=200, return_encoder_output=True),
    dict(return_encoder_output=True),
]
for kw in tries:
    try:
        m = CBraMod.from_pretrained("braindecode/cbramod-pretrained", **kw).eval()
        n = sum(p.numel() for p in m.parameters())
        with torch.no_grad():
            y = m(x)
        print("OK", kw)
        print("parameters:", n, "| output type:", type(y).__name__, "| shape:", getattr(y, "shape", None))
        break
    except Exception as e:
        print("FAILED", kw, "->", type(e).__name__, str(e)[:300])