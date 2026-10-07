"""Methodology, limitations, configuration and project map."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _shared as sh

sh.page_config("About", icon="ℹ")

import streamlit as st  # noqa: E402
import yaml  # noqa: E402

cfg = sh.get_config()
sh.sidebar_status(cfg)

sh.header("About this project",
          "What the system does, how it was built, and - importantly - what it cannot tell you.")

tab_method, tab_limits, tab_config, tab_structure, tab_extend = st.tabs(
    ["Methodology", "Limitations", "Configuration", "Project map", "Extend it"]
)

with tab_method:
    st.markdown(
        """
### Problem

Illegal and unmonitored waste sites are hard to find from the ground and easy to see from
orbit - if someone looks. Sentinel-2 gives free 10 m true-colour imagery of the whole planet
every five days, which is far more imagery than any environmental agency can review by eye.
The job of this system is triage: score every tile, and put a short ranked list of
candidate sites in front of a human.

### Data

**EuroSAT** (Helber et al., 2019) - 27,000 labelled 64x64 RGB chips from Sentinel-2,
covering ten land-cover classes across 34 European countries. It is the standard benchmark
for satellite land-use classification, small enough to train on a laptop CPU, and freely
redistributable.

### The label problem, stated plainly

EuroSAT has no *landfill* class. Rather than invent one, the pipeline trains a clearly
labelled **proxy task**:

| | classes |
| --- | --- |
| positive - waste-site-like | `Industrial`, `Highway` |
| negative - clean land | `Forest`, `Pasture`, `HerbaceousVegetation`, `River`, `SeaLake`, `AnnualCrop`, `PermanentCrop`, `Residential` |

The reasoning: an active landfill is a large, bare, disturbed, high-albedo anthropogenic
surface with haul roads and terraced working faces. In 10 m true colour that signature sits
much closer to industrial hardstanding than to any natural class. Keeping `Residential` on
the *negative* side is deliberate - it forces the model to learn "disturbed bare industrial
surface" rather than the far easier "built-up versus not built-up".

### Preprocessing

A single stratified, seeded **manifest** (`data/interim/manifest.csv`) assigns every image
a task label and a split. Nothing is copied or duplicated, splits are reproducible from the
seed, and the exact composition is auditable. A `tf.data` pipeline then decodes in parallel,
resizes to 128x128, caches, shuffles, batches, and augments the training split with flips,
rotation, zoom, translation, brightness and contrast jitter - the invariances that actually
hold for nadir satellite imagery (there is no canonical "up" in an orbital view, so vertical
flips and full rotations are legitimate here in a way they would not be for street photos).

### Model

ImageNet-pretrained backbone (MobileNetV2 by default) with a fresh head:
global average pooling, dropout, a 256-unit dense layer, batch norm, dropout, sigmoid.
Backbone normalisation lives *inside* the exported model as a `Rescaling` layer, so the
saved `.keras` file takes raw 0-255 pixels and there is no hidden preprocessing contract
between training and serving.

Training runs in two phases:

1. **Warm-up** - backbone frozen, head trained at 1e-3. Stops the randomly initialised head
   from destroying pretrained filters with its large early gradients.
2. **Fine-tuning** - the deepest 40 backbone layers unfrozen at 2e-5, BatchNorm layers kept
   frozen so their running statistics are not corrupted by a small, differently distributed
   dataset.

Class weights, early stopping on validation AUC, LR reduction on plateau and best-checkpoint
restoration are all on by default.

### Explainability

Grad-CAM weights the final feature maps by the gradient of the landfill score, showing which
regions carried the evidence. Because the model keeps backbone and head as separate nested
sub-models, the heatmap is computed by running the two halves in sequence and watching the
feature map in between - robust across Keras versions and free of layer-graph surgery.

### Scene scanning

A tile classifier becomes a crude detector by sliding a window across a full scene with 50%
overlap, averaging the overlapping scores into a per-pixel probability surface, thresholding,
and merging the surviving windows into boxes with a transitive IoU merge. With metres-per-pixel
supplied, each candidate gets an area estimate in hectares.
        """
    )

with tab_limits:
    st.error(
        "**This is a screening aid and a portfolio-grade engineering project. It is not a "
        "validated landfill detector, and no output of it should be used as evidence of "
        "an environmental violation.**",
        icon="⚠",
    )
    st.markdown(
        """
1. **The positive class is a proxy.** Metrics measure how well the model separates
   industrial/road surfaces from natural cover - not landfills from everything else. A
   headline accuracy of 97% on this task does *not* transfer to "97% of landfills found".

2. **Confusable surfaces.** Quarries, construction sites, bare ploughed fields, gravel pits
   and large flat roofs share the signature. The *By land cover* tab on the Model Performance
   page quantifies exactly this, and it is the first thing to read before trusting a result.

3. **Resolution ceiling.** EuroSAT chips are 64x64 at 10 m, so one tile is 640 m across.
   Small dumps, fly-tipping and anything under roughly 30 m simply is not resolvable.

4. **Geography and season.** The training data is European. Arid, tropical and high-latitude
   landscapes have different soil and vegetation spectra; expect the false-alarm rate to rise
   outside Europe and outside the growing season.

5. **RGB only.** The pipeline uses three visible bands. Landfill discrimination improves
   substantially with SWIR (surface moisture and mineralogy), thermal (waste decomposition
   heat) and methane-sensitive bands. EuroSAT's multispectral variant and Sentinel-2's
   13 bands are the obvious upgrade.

6. **No georeferencing by default.** Scene-scan coordinates are pixel offsets. Real
   deployment needs a GeoTIFF reader (`rasterio`) to turn them into latitude and longitude.

7. **Single-moment inference.** Landfills are defined as much by *change over time* as by
   appearance. A time-series model over repeat passes would be far stronger than a
   single-date classifier.

8. **No independent validation set.** Test tiles come from the same distribution as training
   tiles. Real deployment demands a geographically held-out region.
        """
    )

with tab_config:
    st.markdown("Live contents of `config.yaml` - edit the file and refresh the page.")
    st.code(yaml.safe_dump(cfg.raw, sort_keys=False, allow_unicode=True), language="yaml")
    st.caption(f"Loaded from `{cfg.path}`")

with tab_structure:
    st.code(
        """
Landfill Detection Project/
├── config.yaml                  single source of truth for every script
├── requirements.txt
├── setup.bat / setup.sh         one-shot environment setup
├── run_app.bat / run_app.sh     launch the web console
│
├── src/
│   ├── config.py                typed config loader + validation
│   ├── utils.py                 logging, seeding, image and JSON helpers
│   ├── data/
│   │   ├── download.py          EuroSAT acquisition with mirror fallback
│   │   ├── synthetic.py         offline procedural stand-in dataset
│   │   ├── preprocess.py        label mapping, stratified split, manifest
│   │   └── dataset.py           tf.data pipeline + augmentation
│   ├── models/
│   │   ├── build.py             backbone + head, fine-tuning control
│   │   └── gradcam.py           saliency maps and overlays
│   ├── train.py                 two-phase training
│   ├── evaluate.py              metrics, curves, error analysis
│   ├── predict.py               cached inference API
│   └── scan.py                  sliding-window scene detection
│
├── app/
│   ├── Home.py                  overview and dataset composition
│   ├── _shared.py               caching, styling, common widgets
│   └── pages/                   Detect, Scene Scan, Batch, Performance,
│                                Pipeline, About
│
├── scripts/                     01_download_data, 02_train, 03_evaluate,
│                                make_sample_scene, run_pipeline
├── tests/                       pytest suite (runs without a trained model)
│
├── data/     raw, interim, processed, custom, samples   (git-ignored)
├── models/   landfill_detector.keras, model_meta.json
└── reports/  metrics.json, history.csv, figures/, scans/
        """,
        language="text",
    )

with tab_extend:
    st.markdown(
        """
### Train on real landfill labels

1. Collect chips and sort them into two folders:

   ```
   data/custom/landfill/*.jpg
   data/custom/not_landfill/*.jpg
   ```

2. In `config.yaml`:

   ```yaml
   data:
     source: custom
   task:
     positive_classes: ["landfill"]
     negative_classes: ["not_landfill"]
   ```

3. Re-run `python scripts/01_download_data.py` then `python scripts/02_train.py`.

Nothing else changes - the manifest builder, the tf.data pipeline, training, evaluation,
Grad-CAM, scanning and the whole web console pick up the new task automatically.

### Other worthwhile extensions

- **Multispectral input.** Swap in EuroSAT-MS and widen the first convolution to 13 bands;
  SWIR in particular separates bare waste from bare soil.
- **Segmentation.** Replace the classifier head with a U-Net decoder for pixel-level site
  outlines instead of boxes.
- **Change detection.** Stack two dates as a 6-channel input; growth is the strongest single
  signal that a bare patch is an active landfill.
- **Georeferencing.** Read GeoTIFF affine transforms with `rasterio` and emit GeoJSON so
  detections drop straight into QGIS.
- **Calibration.** Fit Platt scaling or isotonic regression on the validation split so the
  reported probabilities are usable as probabilities.
        """
    )

st.divider()
st.caption(
    "Dataset: EuroSAT - Helber, Bischke, Dengel & Borth, *EuroSAT: A Novel Dataset and Deep "
    "Learning Benchmark for Land Use and Land Cover Classification*, IEEE JSTARS, 2019. "
    "Grad-CAM: Selvaraju et al., ICCV 2017."
)
