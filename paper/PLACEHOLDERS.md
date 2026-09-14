# Open placeholders in `main.tex`

Every bracketed uppercase token in `main.tex` marks information that is not yet
established. Nothing in this list may be filled by estimation. Replace a token
only with a value traceable to a run directory, a benchmark measurement, or the
author metadata.

## Author and venue metadata

Author block and affiliations follow the ESGA-Net manuscript, with Nguyen Ngoc
Lan replaced by Giang Tuan Hung:

| # | Author | Affiliations | ORCID |
|---|---|---|---|
| 1 | Le Xuan Thang | 1, 3 | 0000-0002-9911-3544 |
| 2 | Giang Tuan Hung | 1, 3 | not recorded |
| 3 | Vu Manh Trung | 2, 3 | 0009-0009-7596-1520 |
| 4 | Tran Ngoc Hoa (corresponding) | 2, 3 | 0009-0005-9255-3263 |

Affiliation 1 is CIVIX - Civil Intelligence and eXpanded Data Company Limited,
Hanoi, Vietnam; affiliation 2 is the Faculty of Civil Engineering, University of
Transport and Communications, Hanoi, Vietnam; affiliation 3 is the Application of
Artificial Intelligence in Structural Health Monitoring (AI-SHM), University of
Transport and Communications, Hanoi, Vietnam.

| Token | Source |
|---|---|
| `[GTH_EMAIL]` | Giang Tuan Hung |
| `[CODE_REPOSITORY_URL]` | data availability statement |

Two items carried over from ESGA-Net that this manuscript does not yet have: an
ORCID for Giang Tuan Hung, and `\credit{}` CRediT statements with `\printcredits`.
Add both if the target journal requires them.

The class is `cas-sc` (single column). Switch to `cas-dc` if the target journal
requires the double-column layout. Citations are IEEE numbered
(`\usepackage[numbers,sort&compress]{natbib}` with `\bibliographystyle{IEEEtran}`,
and `\cite{}` throughout); `IEEEtran.bst` is kept in this directory so the build
is self-contained.

## Dataset facts

| Token | Source |
|---|---|
| `[MASS_PAIRS]`, `[MASS_TRAIN]` | count entries in the Massachusetts `train.txt` / `test.txt` lists |

The DeepGlobe counts (6226 / 5000 / 300 / 1226) and the Massachusetts
validation and test counts (61 / 117) are fixed by the split code and need no
verification.

## Training configuration

| Token | Source |
|---|---|
| `[HARDWARE]` | GPU model and count used for the reported runs |
| `[PYTORCH_VERSION]`, `[CUDA_VERSION]` | training environment |
| `[MASS_EPOCHS]`, `[DG_EPOCHS]` | `--epochs` actually used |
| `[NUMBER_OF_SEEDS]` | number of independent runs behind each reported mean |

## Results

All `[VAL]` cells in Tables 3, 4, and 5 (`tab:sota`, `tab:ablation`,
`tab:efficiency`), plus:

| Token | Source |
|---|---|
| `[IOU_MASS]`, `[IOU_DG]` | `test_native.py` on `test117` and `deepglobe_test1226` |
| `[DELTA_MASS]`, `[DELTA_DG]` | absolute difference in percentage points against the strongest baseline |
| `[GATE_MEAN]`, `[GATE_STD]` | `gate_statistics()` on the final checkpoint |
| `[ABLATION_DATASET]` | which benchmark the ablation table reports |
| `[BENCHMARK_RUNS]`, `[WARMUP_RUNS]`, `[BENCHMARK_HARDWARE]` | `compare_reparameterization.py` |
| latency and peak VRAM in Table 4 | `compare_reparameterization.py` |

Bracketed *paragraph* directives (`[RESULTS PARAGRAPH: ...]`,
`[ABLATION PARAGRAPH, GATE: ...]`, and the rest) are drafting instructions, not
text. Each states what the paragraph must establish once the numbers exist.

Three `[VERIFY: ...]` notes mark claims that must be checked before submission:
the DeepGlobe protocol used by the comparison methods, rerunning the
re-parameterization equivalence check on the final trained checkpoints, and the
draft CRediT role assignment, which each author must confirm for themselves.

## Figures

`figs/` is empty. Two figures are referenced through the `\placeholderfig`
macro, which draws a labeled box so the document still compiles:

- `figs/architecture.pdf` (Fig. 1)
- `figs/qualitative.pdf` (Fig. 2)

Replace each `\placeholderfig{...}` call with `\includegraphics` once the file
exists, and record the analysis that generated it.

## Values already verified (do not treat as placeholders)

These were measured from the code in this repository and are stated directly in
the manuscript:

| Quantity | Value |
|---|---|
| Total parameters, training form | 22,541,845 |
| Total parameters, after `switch_to_deploy()` | 22,502,773 |
| ResNet-34 encoder parameters | 21,284,672 |
| Spatial-gate parameters (spatial minus static variant) | 232,050 |
| Parameters beyond the encoder | 1,257,173 (5.6%) |
| Detail stream (projection + 4 RepVGG blocks) | 383,424 |
| Bilateral exchange projections | 246,464 |
| ProgressiveDAPPM with its projections | 284,864 |
| ControlledRoadFusion with its projection | 55,872 |
| Per-channel residual scales | 608 |
| Reconstruction decoder | 53,891 (incl. 18,529 centerline head) |
| GMACs at 1024x1024, training form | 90.90 |
| GMACs at 1024x1024, deploy form | 90.46 |
| Max abs logit difference, fused vs. multi-branch (untrained, 512x512, fp32) | 1.2e-7 |
| Max abs error, `RepVGGBlock` fusion | 9.5e-6 |
| Max abs error, `RepDepthwiseBlock` fusion | 9.5e-7 |

The static (ungated) variant has 22,309,795 parameters and 88.94 GMACs, which is
the number to use for the ablation row if that configuration is retrained.

## Build

```bash
latexmk -pdf main.tex
```

The single overfull `\hbox` warning at `\maketitle` originates in `cas-sc.cls`
and appears identically in the unmodified upstream template.
