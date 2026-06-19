# Day 1–2 — Pathology AI review

> Block 1 of the two-week plan. Reading goal (mentor §3): understand what the patch encoders in this project actually do. Facts below were web-verified on 2026-06-18 against the primary sources cited at the bottom.

## 1. What is Computational Pathology doing?

Pathology = diagnosing disease by looking at stained tissue under a microscope;
it is the **gold standard** for cancer diagnosis. Computational pathology
digitizes this:

1. **Digitization** — a slide scanner turns a glass slide into a **Whole-Slide
  Image (WSI)**: a gigapixel image (often ~100,000² px, several GB).
2. **AI analysis** — models do tumor detection, grading, subtyping, prognosis,
  and even prediction of *molecular* properties (mutations, gene expression).

The field's recent shift is to **pathology foundation models**: pretrain a
general feature extractor on millions of unlabeled WSIs via self-supervision,
then transfer to many downstream tasks with little labeled data. The premise
behind HEST: **morphology encodes molecular state** (genes → proteins → cell
behavior → appearance), so a strong encoder should expose signal that predicts
gene expression.

## 2. The basic data objects

- **H&E staining** — Hematoxylin stains **nuclei** blue/purple; Eosin stains
**cytoplasm / stroma** pink. The standard routine stain.
- **WSI** — the gigapixel digital slide.
- **Magnification / MPP** — "20×" ≈ **0.5 µm/px** (MPP = microns per pixel).
Patches must be resampled to a common physical resolution to be comparable.
- **Tiling / patching** — WSIs are too big for a network, so they are cut into
small **patches** (here 256×256 px) that the encoder processes one at a time.
- **Spatial Transcriptomics (ST)** — measures gene expression **with spatial
location preserved** (e.g. 10x Visium). A **spot** is one measurement location
(Visium spot ≈ 55 µm) yielding an expression vector over ~20k genes. Because
the H&E image and the ST grid cover the **same tissue**, each spot maps to a
pixel location → the morphology↔expression pairing HEST exploits.

## 3. Patch feature extractor — what it is

A **patch encoder** maps an image patch (256×256×3) → a fixed-length
**embedding** (a vector). Similar morphology ⇒ nearby embeddings. The benchmark
freezes the encoder and probes the embeddings with a linear model, so it
measures **the quality of the features themselves**, not a downstream network.

## 4. Three training paradigms (mentor §三)

**(a) ImageNet-pretrained CNN (baseline).** E.g. ResNet-50 trained on natural
images (cats, cars). Transfers poorly to histology — the HEST ResNet-50 baseline
(~0.325 avg) sits at the bottom of the leaderboard.

**(b) Pathology self-supervised pretraining.** Two members used here:


| Model        | Paradigm                                              | Backbone                       | Pretrain data                                               | Params                       | Source                                                 |
| ------------ | ----------------------------------------------------- | ------------------------------ | ----------------------------------------------------------- | ---------------------------- | ------------------------------------------------------ |
| **CONCH**    | Vision–**language** contrastive + captioning (CoCa)   | ViT-B/16 vision + text decoder | **1.17M** histo **image–caption** pairs (PMC-OA + internal) | ~200M (90M vision + 110M LM) | Lu et al., *Nat. Med.* 2024                            |
| **Virchow2** | **Vision-only** self-supervised (modified **DINOv2**) | **ViT-H/14**                   | **3.1M WSIs** (MSKCC), mixed magnification 5/10/20/40×      | **632M**                     | Vorontsov et al., arXiv 2408.00738 (Paige + MSR), 2024 |


- **CONCH** ("CONtrastive learning from Captions for Histopathology"): learns by
aligning images with the text that describes them (figure captions / reports),
giving semantically meaningful features. For patch encoding only the **vision
tower (ViT-B/16, 512-d output)** is used. This project's "CONCH" = `conch_v1`,
the original Table-1 entry.
- **Virchow2**: a ViT-H trained with a **modified DINOv2** objective (KoLeo
regularizer → kernel density estimator; crop-and-resize → "extended context
translation"), on mixed-magnification tiles. Much larger than CONCH → richer
but heavier (needs ~40 GB ideally; we ran it at batch_size 8 on 12 GB). Output
dim ~2560.
- (Context) **UNI** (Mahmood Lab, *Nat. Med.* 2024): ViT-L/16 via DINOv2 on
"Mass-100K" (~100M patches from ~100k WSIs); **UNI2-h** (Jan 2025) scales to
  > 200M images / 350k+ WSIs. UNIv2 is the backbone behind UNIv2-SEAL.

**(c) ST-molecular fine-tuning (SEAL — our bonus).** Takes a frozen foundation
backbone (CONCH or UNIv2) and **injects spatial-transcriptomic signal** via
parameter-efficient fine-tuning (LoRA adapters), making morphology features more
"molecule-aware". See `day13-14_seal_bonus.md`.

## 5. DINOv2 in one paragraph (the engine behind Virchow2 / UNI)

**DINOv2 = "self-DIstillation with NO labels."** A **student** ViT learns to
match a **teacher** ViT whose weights are an **exponential moving average** of
the student's (no pretrained teacher, no labels). Two objectives at different
granularities: **image-level (DINO)** — class tokens of two crops must agree via
prototype scores; **patch-level (iBOT)** — the student sees masked patches and
must predict the teacher's representation of the unmasked ones. Meta trained the
original on 142M natural images; pathology models reuse the recipe on WSIs.

## 6. Why this matters for the task

The whole HEST-Benchmark question is: *which of these encoders produces features
from which spot-level gene expression is most linearly predictable?* That is why
the protocol freezes the encoder, reduces dim with PCA(256) for fairness, and
fits a simple Ridge probe — so the score reflects the **encoder**, not the head.

## Open questions / things confirmed

- CONCH default = v1 (Table-1 entry); v1.5 also wired in. Confirmed.
- Virchow2 VRAM: paper-scale runs use big GPUs; batch_size 8 works on 12 GB.
- Reference-magnitude discrepancy (mentor §八 vs actual HEST values) → see
`day12_comparison.md`.

## Sources

- Song, Jaume, Williamson et al., "Artificial intelligence for digital and
computational pathology", *Nature Reviews Bioengineering* 1:930–949 (2023),
DOI 10.1038/s44222-023-00096-8 (preprint arXiv:2401.06148).
- Lu et al., "A visual-language foundation model for computational pathology",
*Nature Medicine* 2024 (s41591-024-02856-4); arXiv:2307.12914. CONCH repo:
github.com/mahmoodlab/CONCH.
- Vorontsov et al., "Virchow2: Scaling Self-Supervised Mixed Magnification
Models in Pathology", arXiv:2408.00738 (2024); hf.co/paige-ai/Virchow2.
- Chen et al., "Towards a general-purpose foundation model for computational
pathology" (UNI), *Nature Medicine* 2024; hf.co/MahmoodLab/UNI.
- Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision",
Meta AI, 2023.

