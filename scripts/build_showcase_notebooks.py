"""Create and execute the three lightweight test-set showcase notebooks."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"

DATASETS = {
    "cub": {
        "title": "CUB-200-2011 birds",
        "results": "../results/cub/RESULTS.md",
        "models": (
            ("ResNet-50", "resnet50"),
            ("DenseNet-121", "densenet121"),
            ("EfficientNet-B0", "efficientnet-b0"),
            ("ViT-Tiny/16", "vit-tiny-p16-avg"),
            ("ViT-Small/16", "vit-small-p16-avg"),
            ("ViT-Base/16", "vit-base-p16-avg"),
        ),
    },
    "isic": {
        "title": "ISIC skin lesions",
        "results": "../results/isic/RESULTS.md",
        "models": (
            ("ResNet-50", "resnet50"),
            ("DenseNet-121", "densenet121"),
            ("EfficientNet-B0", "efficientnet-b0"),
            ("ViT-Tiny/16", "vit-tiny-p16-avg"),
            ("ViT-Small/16", "vit-small-p16-avg"),
            ("ViT-Base/16", "vit-base-p16-avg"),
        ),
    },
    "oxford_pet": {
        "title": "Oxford-IIIT Pet",
        "results": "../results/oxford_pet/RESULTS.md",
        "models": (
            ("ResNet-50", "resnet50"),
            ("DenseNet-121", "densenet121"),
            ("EfficientNet-B0", "efficientnet-b0"),
            ("ViT-Tiny/16", "vit-tiny-p16-avg"),
            ("ViT-Small/16", "vit-small-p16-avg"),
            ("ViT-Base/16", "vit-base-p16-avg"),
        ),
    },
}


def markdown(text: str):
    return nbformat.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbformat.v4.new_code_cell(text.strip())


def make_notebook(key: str, metadata: dict) -> nbformat.NotebookNode:
    cells = [
        markdown(
            f"""
# {metadata['title']} — test-set CAM audit

This notebook is the visual companion to the [full benchmark]({metadata['results']}).
It shows **three frozen test examples per architecture** using the exact same
center-cropped input for CE, FCAL V1, FCAL V2, and—where available—Universal FCAL.

How to read each panel:

- green/red text means the predicted class is correct/incorrect;
- the white contour is the annotated foreground;
- `inside` is the share of positive **predicted-class CAM** mass inside that contour;
- CAM colors are normalized within each panel and use `jet` (blue = low,
  red = high).

The examples are selected reproducibly from saved test arrays: first a strong
localization comparison where CE and V2 are both correct, then a CE→V2
classification rescue when one exists, then another localization improvement.
This is a qualitative audit, not an accuracy estimate; use the full result tables
for quantitative claims.
"""
        ),
        code(
            f"""
from pathlib import Path
import pandas as pd

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent

ASSETS = ROOT / "assets" / "result_showcases" / "{key}"
manifest = pd.read_csv(ASSETS / "examples.csv")
print(f"Loaded {{len(manifest)}} frozen examples from {{manifest['architecture'].nunique()}} architectures")
manifest[["family", "architecture", "case_rank", "image_id", "true_class"]]
"""
        ),
        markdown(
            """
## CNNs

These CAMs are exact final-feature class activation maps: the classifier's own
linear weights projected back onto the final convolutional feature grid.
"""
        ),
    ]

    for title, stem in metadata["models"][:3]:
        cells.append(markdown(f"### {title}\n\n![{title} test cases](../assets/result_showcases/{key}/{stem}_test_cases.png)"))

    cells.append(
        markdown(
            """
## Vision Transformers

The ViT runs use average-pooled patch tokens, so their class-specific patch CAMs
decompose the actual classifier score rather than substituting a generic
self-attention visualization.
"""
        )
    )
    for title, stem in metadata["models"][3:]:
        cells.append(markdown(f"### {title}\n\n![{title} test cases](../assets/result_showcases/{key}/{stem}_test_cases.png)"))

    cells.extend(
        [
            markdown(
                """
## Selected-example audit table

The table below exposes every displayed prediction and inside-foreground value.
Because these examples were deliberately selected for visual diagnosis, do not
average this table or present it as benchmark performance.
"""
            ),
            code(
                """
display_columns = [
    column for column in manifest.columns
    if column in {"architecture", "case_rank", "image_id", "true_class"}
    or column.endswith("_correct")
    or column.endswith("_inside_fraction")
]
manifest[display_columns].style.format({
    column: "{:.1%}" for column in display_columns if column.endswith("_inside_fraction")
})
"""
            ),
            markdown(
                """
## Artifact provenance

The committed PNGs were reconstructed from the real best-validation checkpoints.
The accompanying `examples.csv` manifest freezes every displayed test ID,
prediction, correctness flag, confidence, and inside-foreground CAM value, so
this notebook remains auditable without downloading checkpoints or datasets.
"""
            ),
        ]
    )
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {
        "display_name": "Python 3", "language": "python", "name": "python3",
    }
    notebook.metadata.language_info = {"name": "python", "version": "3.10"}
    return notebook


def main() -> None:
    NOTEBOOKS.mkdir(exist_ok=True)
    for key, metadata in DATASETS.items():
        path = NOTEBOOKS / f"{key}_test_showcase.ipynb"
        notebook = make_notebook(key, metadata)
        NotebookClient(
            notebook,
            timeout=60,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        ).execute()
        nbformat.write(notebook, path)
        print(f"Wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
