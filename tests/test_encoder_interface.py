"""Tests for the SEAL encoder wrapper's Trident-style interface.

The real SEAL model load (and its gated weights) is mocked: we inject a fake
``seal`` module whose ``seal_factory`` returns a tiny dummy model. This lets us
verify the wrapper's contract — that ``forward`` produces ``[B, D]`` embeddings
and that ``eval_transforms`` / ``precision`` are exposed — without any weights,
network, or GPU.
"""

import os
import sys
import types

import pytest

torch = pytest.importorskip("torch")  # needs torch (in environment.yml)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EMBED_DIM = 512  # native CONCH dim; the dummy model mimics SEAL's vision output.


class _DummyVisionModel(torch.nn.Module):
    """Stand-in for SEAL's PatchRecEncoder: maps an image batch to [B, D]."""

    def __init__(self, dim=EMBED_DIM):
        super().__init__()
        self.dim = dim
        # A trivial learnable param so .to()/.eval()/parameters() behave.
        self.proj = torch.nn.Linear(3, dim)

    def forward(self, x):  # x: [B, C, H, W]
        b = x.shape[0]
        pooled = x.mean(dim=(2, 3))           # [B, C]
        return self.proj(pooled)              # [B, D]


def _install_fake_seal(monkeypatch, dim=EMBED_DIM):
    """Insert a fake `seal` module exposing seal_factory into sys.modules."""
    fake = types.ModuleType("seal")

    def seal_factory(backbone, source="auto", hf_token=None,
                     gene_post_warmup=False, **kwargs):
        model = _DummyVisionModel(dim)
        transforms = lambda img: img  # noqa: E731 - identity transform placeholder
        precision = torch.float16
        gene_model = object()
        return (model, transforms, precision), gene_model

    fake.seal_factory = seal_factory
    monkeypatch.setitem(sys.modules, "seal", fake)
    return fake


@pytest.fixture(autouse=True)
def _fresh_encoder_module():
    """Ensure the encoder module is re-imported per test (clean trident state)."""
    sys.modules.pop("src.encoders.conch_seal_encoder", None)
    yield
    sys.modules.pop("src.encoders.conch_seal_encoder", None)


def test_conch_seal_forward_shape(monkeypatch):
    _install_fake_seal(monkeypatch)
    from src.encoders.conch_seal_encoder import CONCHSEALEncoder

    enc = CONCHSEALEncoder()
    enc.eval()

    batch = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        out = enc(batch)

    assert out.shape == (4, EMBED_DIM)            # [B, D]
    assert enc.enc_name == "conch_seal"
    # Trident-style attributes must be present for HEST's benchmark() to accept it.
    assert hasattr(enc, "eval_transforms")
    assert enc.precision == torch.float16


def test_univ2_seal_backbone_name(monkeypatch):
    _install_fake_seal(monkeypatch, dim=1536)
    from src.encoders.conch_seal_encoder import UNIv2SEALEncoder

    enc = UNIv2SEALEncoder()
    out = enc(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 1536)
    assert enc.backbone == "univ2"
    assert enc.enc_name == "univ2_seal"


def test_build_seal_encoder_registry(monkeypatch):
    _install_fake_seal(monkeypatch)
    from src.encoders.conch_seal_encoder import build_seal_encoder, CONCHSEALEncoder

    enc = build_seal_encoder("conch_seal")
    assert isinstance(enc, CONCHSEALEncoder)

    with pytest.raises(ValueError):
        build_seal_encoder("does_not_exist")


def test_missing_vision_model_raises(monkeypatch):
    """If seal_factory returns no vision model, _build should raise clearly."""
    fake = types.ModuleType("seal")

    def seal_factory(backbone, **kwargs):
        return (None, None, None), None

    fake.seal_factory = seal_factory
    monkeypatch.setitem(sys.modules, "seal", fake)

    from src.encoders.conch_seal_encoder import CONCHSEALEncoder
    with pytest.raises(FileNotFoundError):
        CONCHSEALEncoder()
