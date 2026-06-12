"""SEAL patch encoders wrapped in Trident's ``BasePatchEncoder`` interface.

SEAL (mahmoodlab/SEAL) fine-tunes a pathology foundation-model backbone with
spatial-transcriptomics signal. Its public loader returns a vision encoder that
behaves like any other patch encoder, so we can wrap it to look exactly like a
native Trident encoder and feed it to either:

  * Trident's ``Processor.run_patch_feature_extraction_job(patch_encoder=...)``, or
  * HEST's ``benchmark(encoder, None, None, ...)`` (which accepts any object
    exposing ``.eval_transforms`` and ``.precision``).

Real interfaces this is modelled on (read from the cloned repos, not guessed):

  Trident  third_party/trident/trident/patch_encoder_models/load.py
    class BasePatchEncoder(torch.nn.Module):
        def __init__(self, weights_path=None, **build_kwargs):
            ...
            self.model, self.eval_transforms, self.precision = self._build(**build_kwargs)
        def forward(self, x): return self.model(x)            # overridable
        @abstractmethod
        def _build(self, **kw) -> (nn.Module, Callable, torch.dtype): ...

  SEAL     third_party/SEAL/seal/models/encoder_factory.py
    def seal_factory(backbone, source="auto", hf_repo_id="MahmoodLab/SEAL",
                     hf_revision="main", hf_token=None, ...):
        # returns ((img_model, img_transforms, img_precision), gene_model)

Backbone embedding dims (native, pre-SEAL-projection; see SEAL EMB_DICT):
    conch  -> 512    (ViT-B/16)
    univ2  -> 1536   (ViT-H/14)
SEAL's vision head may project these to its own ``out_dim``; the benchmark's PCA
step (256 factors) sits downstream and is dimension-agnostic.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

try:  # Prefer the real Trident base class on the GPU machine.
    import torch
    from trident.patch_encoder_models.load import BasePatchEncoder

    _HAS_TRIDENT = True
except Exception:  # pragma: no cover - exercised only where trident is absent
    # Fallback shim so this module imports (and the forward-shape unit test runs)
    # on a laptop without Trident installed. It mirrors the real
    # BasePatchEncoder contract above 1:1. On the GPU box the real class is used.
    _HAS_TRIDENT = False
    try:
        import torch
    except Exception:  # torch genuinely missing — let import fail loudly later.
        torch = None  # type: ignore

    _Base = torch.nn.Module if torch is not None else object

    class BasePatchEncoder(_Base):  # type: ignore[misc, valid-type]
        """Minimal stand-in for trident...load.BasePatchEncoder."""

        def __init__(self, weights_path: Optional[str] = None, **build_kwargs: Any):
            super().__init__()
            self.enc_name: Optional[str] = None
            self.weights_path: Optional[str] = weights_path
            self.model, self.eval_transforms, self.precision = self._build(**build_kwargs)

        def forward(self, x):  # noqa: D401 - identical default to Trident
            return self.model(x)

        def _build(self, **build_kwargs: Any):
            raise NotImplementedError


class SEALPatchEncoder(BasePatchEncoder):
    """Base wrapper around ``seal_factory`` for any supported SEAL backbone.

    Args (passed to ``__init__``, not ``_build``, mirroring how Trident's own
    encoders stash config on ``self`` before ``super().__init__`` runs):
        backbone: SEAL backbone key. HF-hosted options: ``"conch"``, ``"univ2"``.
        hf_token: Hugging Face token; falls back to ``HF_TOKEN`` env var inside
            ``seal_factory`` when ``None``.
        source: ``"auto"`` (HF then local), ``"hf"`` (HF only) or ``"local"``
            (read from ``weights/{backbone}_SEAL/``).
        gene_post_warmup: forwarded to ``seal_factory`` (selects which omics
            checkpoint variant to load; unused for pure feature extraction).
    """

    BACKBONE = "conch"  # subclasses override

    def __init__(
        self,
        backbone: Optional[str] = None,
        hf_token: Optional[str] = None,
        source: str = "auto",
        gene_post_warmup: bool = False,
        weights_path: Optional[str] = None,
        **build_kwargs: Any,
    ):
        self.backbone = backbone or self.BACKBONE
        self.hf_token = hf_token
        self.source = source
        self.gene_post_warmup = gene_post_warmup
        # Keep a handle on the SEAL gene/omics model in case downstream code
        # wants image<->gene retrieval; not used for plain feature extraction.
        self.gene_model = None
        super().__init__(weights_path=weights_path, **build_kwargs)

    def _build(self, **build_kwargs: Dict[str, Any]) -> Tuple[Any, Callable, Any]:
        """Load the SEAL vision encoder via ``seal_factory``.

        Returns ``(model, eval_transforms, precision)`` as required by
        ``BasePatchEncoder``.
        """
        # NOTE: import is local so the module imports without `seal` installed
        # (e.g. on the no-GPU scaffolding machine, and so tests can monkeypatch).
        try:
            from seal import seal_factory
        except Exception as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "Could not import `seal`. Install SEAL on the GPU machine via "
                "`pip install -e third_party/SEAL` (see scripts/setup_env.sh)."
            ) from exc

        self.enc_name = f"{self.backbone}_seal"

        # TODO(weights): SEAL checkpoints are GATED on Hugging Face
        #   (repo: MahmoodLab/SEAL). Before this runs end-to-end you must:
        #     1. Request access at https://huggingface.co/MahmoodLab/SEAL
        #     2. Set HF_TOKEN (see .env / .env.example), or pass hf_token=...
        #   Expected checkpoint filenames for backbone `conch`:
        #     - seal_conch_vision.pth
        #     - seal_conch_omics.pth
        #   With source="auto"/"hf" seal_factory downloads them to
        #   weights/{backbone}_SEAL/; with source="local" it reads them from
        #   that directory (download manually first — see README / SEAL README).
        #   Until access is granted seal_factory will raise; that is expected.
        (img_model, img_transforms, img_precision), gene_model = seal_factory(
            backbone=self.backbone,
            source=self.source,
            hf_token=self.hf_token,
            gene_post_warmup=self.gene_post_warmup,
        )
        self.gene_model = gene_model

        if img_model is None:
            # seal_factory returns None when no checkpoint was found.
            raise FileNotFoundError(
                f"SEAL returned no vision model for backbone '{self.backbone}'. "
                "Most likely the gated weights are not yet downloaded/authorised. "
                "See the TODO in SEALPatchEncoder._build."
            )

        return img_model, img_transforms, img_precision

    def forward(self, x):
        """Return patch embeddings of shape ``[B, D]``.

        SEAL's vision wrapper (``PatchRecEncoder``) already returns the pooled
        embedding from its ``forward``, so the Trident default (``self.model(x)``)
        is correct. Kept explicit for clarity and to document the contract.
        """
        return self.model(x)


class CONCHSEALEncoder(SEALPatchEncoder):
    """CONCH-SEAL: SEAL fine-tuned on a CONCH (ViT-B/16) backbone.

    Native CONCH embedding dim is 512 (see SEAL EMB_DICT).
    """

    BACKBONE = "conch"


class UNIv2SEALEncoder(SEALPatchEncoder):
    """UNIv2-SEAL: SEAL fine-tuned on a UNIv2 (ViT-H/14) backbone.

    Native UNIv2 embedding dim is 1536 (see SEAL EMB_DICT).
    """

    BACKBONE = "univ2"


# Convenience registry mirroring Trident's `encoder_registry` style, so callers
# can do `SEAL_ENCODERS["conch_seal"]()` symmetrically with Trident encoders.
SEAL_ENCODERS = {
    "conch_seal": CONCHSEALEncoder,
    "univ2_seal": UNIv2SEALEncoder,
}


def build_seal_encoder(name: str, **kwargs: Any) -> SEALPatchEncoder:
    """Instantiate a SEAL encoder by short name (``conch_seal`` / ``univ2_seal``)."""
    if name not in SEAL_ENCODERS:
        raise ValueError(
            f"Unknown SEAL encoder '{name}'. Options: {sorted(SEAL_ENCODERS)}"
        )
    return SEAL_ENCODERS[name](**kwargs)
