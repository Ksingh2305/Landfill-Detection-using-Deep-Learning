# Landfill Detection from Satellite Imagery

A complete, runnable computer-vision system that screens satellite imagery for
waste-site-like surfaces: **scalable preprocessing → transfer-learned deep
classifier → Grad-CAM explanations → sliding-window scanning of full scenes →
a web console you run locally in your browser.**

Built with TensorFlow/Keras and Streamlit. Trains on a laptop CPU in minutes.
No GPU required, no cloud account, no API keys.

---

## Quick start (Windows)

Double-click these in order, from this folder:

| # | File | What it does | Time |
|---|------|--------------|------|
| 1 | `setup.bat` | Creates `.venv` and installs TensorFlow, Streamlit and friends | 5–10 min |
| 2 | `run_pipeline.bat` | Downloads EuroSAT (~90 MB), trains, evaluates | 20–60 min |
| 3 | `run_app.bat` | Opens the web console at http://localhost:8501 | instant |

**In a hurry?** `run_pipeline.bat --quick` uses 300 images per class and 2+2
epochs — usually under five minutes on a modern laptop.

**No internet, or the download is blocked?** `run_pipeline.bat --quick --synthetic`
generates a procedural stand-in dataset locally and trains on that. Everything
works; only the science is meaningless.

### macOS / Linux

```bash
chmod +x setup.sh run_pipeline.sh run_app.sh
./setup.sh
./run_pipeline.sh --quick
./run_app.sh
```

### Doing it by hand

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt

python scripts/01_download_data.py      # acquire imagery + build the manifest
python scripts/02_train.py              # two-phase transfer learning
python scripts/03_evaluate.py           # metrics, curves, error analysis
python scripts/make_sample_scene.py     # a demo scene for the scan page

streamlit run app/Home.py               # the web console
```

> Every step is optional from the browser too — the **Pipeline** page in the
> app runs the same scripts and streams their output live.

---

## The web console

Six pages, all served locally at `localhost:8501`:

| Page | What you can do |
|------|-----------------|
| **Home** | Model status, dataset composition, how the pipeline fits together |
| **Detect** | Drop in one chip → verdict, class probabilities, Grad-CAM heatmap, distance from the decision boundary |
| **Scene Scan** | Upload a full scene → sliding-window sweep, probability surface, merged candidate boxes with hectare estimates, downloadable JSON/CSV/PNG |
| **Batch Analysis** | Score a folder, a zip or a pile of uploads → ranked table, flag rate, galleries of the highest and lowest scoring chips, CSV export |
| **Model Performance** | Confusion matrix, ROC/PR curves, per-land-cover error analysis, interactive threshold chooser, training curves, misclassification browser |
| **Pipeline** | Run data acquisition, training and evaluation from the browser with live logs |
| **About** | Full methodology, limitations, live config, project map, extension guide |

---

## What the model actually detects — read this

EuroSAT, the dataset this trains on, **has no landfill class**. Rather than
pretend otherwise, the project trains a clearly labelled **proxy task**:

| | land covers |
|---|---|
| **positive** — waste-site-like | `Industrial`, `Highway` |
| **negative** — clean land | `Forest`, `Pasture`, `HerbaceousVegetation`, `River`, `SeaLake`, `AnnualCrop`, `PermanentCrop`, `Residential` |

An active landfill is a large, bare, disturbed, high-albedo anthropogenic
surface with haul roads and terraced working faces. At 10 m true colour that
signature sits far closer to industrial hardstanding than to any natural class.
Keeping `Residential` on the *negative* side is deliberate — it stops the model
from solving the much easier "built-up vs. not built-up" problem instead.

**So:** this gives you a real, reproducible satellite-imagery detector and a
feature extractor that transfers well. It does **not** give you a validated
landfill classifier, and no output should be treated as evidence of an
environmental violation. `reports/METHODOLOGY.md` and the app's About page
spell out every limitation.

**To train on real landfill labels**, drop your chips into
`data/custom/landfill/` and `data/custom/not_landfill/`, set

```yaml
data:
  source: custom
task:
  positive_classes: ["landfill"]
  negative_classes: ["not_landfill"]
```

in `config.yaml`, and re-run steps 1 and 2. Nothing else changes.

---

## How it works

### 1. Acquisition — `src/data/download.py`

EuroSAT (Helber et al., 2019): 27,000 labelled 64×64 RGB Sentinel-2 chips over
ten land-cover classes. Downloaded once from a list of mirrors, tried in order,
with size and zip-integrity checks; the archive is normalised into
`data/raw/EuroSAT/<Class>/*.jpg` whatever nesting the mirror used. If every
mirror fails, `--synthetic` generates a procedural stand-in so the pipeline is
never blocked.

### 2. Preprocessing — `src/data/preprocess.py`, `src/data/dataset.py`

Instead of copying tens of thousands of files into `train/val/test` folders, the
pipeline builds a **manifest**: one CSV row per image with its resolved task
label and split assignment. Splits are stratified on the *source* land-cover
class (so each split keeps the same land-cover mix), seeded, reproducible, and
auditable — and regenerating them costs nothing.

The `tf.data` pipeline then decodes in parallel, resizes to 128×128, caches,
shuffles, batches, and augments the training split only: flips, rotation, zoom,
translation, brightness and contrast jitter. Full rotations and vertical flips
are legitimate here in a way they are not for ordinary photographs — a nadir
satellite view has no canonical "up".

### 3. Model — `src/models/build.py`

```
image (0–255)  →  Rescaling  →  backbone  →  head  →  P(landfill)
```

- **Backbone**: MobileNetV2 by default (EfficientNetB0 and ResNet50V2 also
  supported), ImageNet weights.
- **Head**: global average pooling → dropout → dense 256 + ReLU → batch norm →
  dropout → sigmoid.
- Backbone normalisation lives **inside** the exported model, so the saved
  `.keras` file takes raw pixels — no hidden preprocessing contract between
  training and serving.
- `backbone` and `head` are kept as *named nested models*, which is what makes
  Grad-CAM robust across Keras versions.

### 4. Training — `src/train.py`

Two phases, because fine-tuning a pretrained network with a randomly
initialised head attached destroys the pretrained filters:

1. **Warm-up** — backbone frozen, head only, lr 1e-3.
2. **Fine-tuning** — deepest 40 backbone layers unfrozen, lr 2e-5, BatchNorm
   layers kept frozen so their running statistics survive contact with a small,
   differently distributed dataset.

Inverse-frequency class weights, early stopping on validation AUC, LR reduction
on plateau, best-checkpoint restore and a CSV log of every epoch.

### 5. Evaluation — `src/evaluate.py`

Accuracy, balanced accuracy, macro F1, MCC, Cohen's κ, ROC AUC, average
precision, miss rate and false-alarm rate; confusion matrix; ROC and PR curves;
a full threshold sweep; a gallery of the most confident mistakes; and — the
useful one — **recall broken down by original land cover**, which tells you
exactly which surfaces trigger false alarms.

### 6. Explainability — `src/models/gradcam.py`

Grad-CAM weights the backbone's final feature maps by the gradient of the
landfill score. On a true positive you should see the bare working face and haul
roads light up; if the heat sits on a car park or the image border, the model is
keying on a spurious cue and that call should be distrusted.

### 7. Scene scanning — `src/scan.py`

A tile classifier becomes a crude detector: sweep a window across the scene with
50% overlap, average overlapping scores into a per-pixel probability surface,
threshold it, and merge surviving windows into boxes with a transitive IoU
merge. Supply metres-per-pixel and each candidate gets an area in hectares.

---

## Project structure

```
Landfill Detection Project/
├── config.yaml                  single source of truth for every script
├── requirements.txt
├── setup.bat / setup.sh
├── run_pipeline.bat / .sh
├── run_app.bat / .sh
│
├── src/
│   ├── config.py                typed config loader with validation
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
│   ├── Home.py                  streamlit entry point
│   ├── _shared.py               caching, styling, common widgets
│   ├── assets/style.css
│   └── pages/                   Detect, Scene Scan, Batch, Performance,
│                                Pipeline, About
│
├── scripts/
│   ├── 01_download_data.py
│   ├── 02_train.py
│   ├── 03_evaluate.py
│   ├── make_sample_scene.py
│   └── run_pipeline.py          all of the above, in order
│
├── tests/                       pytest suite (no trained model needed)
│
├── data/      raw · interim · processed · custom · samples   (git-ignored)
├── models/    landfill_detector.keras · model_meta.json
└── reports/   metrics.json · history.csv · figures/ · scans/ · METHODOLOGY.md
```

---

## Configuration

Everything is driven by `config.yaml` — no constants buried in code. The knobs
you are most likely to touch:

```yaml
data:
  image_size: [128, 128]     # network input; larger = slower but more detail
  batch_size: 32             # drop to 8–16 if you run out of RAM
  cache: true                # set false on machines with < 8 GB RAM
  max_per_class: 0           # 0 = everything; 300 for a fast smoke run

task:
  positive_classes: [Industrial, Highway]
  threshold: 0.50            # set this from the Model Performance page

model:
  backbone: mobilenetv2      # mobilenetv2 | efficientnetb0 | resnet50v2

train:
  epochs_head: 8
  epochs_finetune: 12
  unfreeze_last_n: 40
  mixed_precision: false     # true only on an NVIDIA GPU

scan:
  tile_size: 128
  overlap: 0.5
  min_confidence: 0.60
```

---

## Command-line reference

```bash
# Data
python scripts/01_download_data.py                       # EuroSAT
python scripts/01_download_data.py --synthetic           # offline stand-in
python scripts/01_download_data.py --max-per-class 300   # small manifest
python scripts/01_download_data.py --force               # re-download

# Training
python scripts/02_train.py
python scripts/02_train.py --backbone efficientnetb0 --batch-size 16
python scripts/02_train.py --epochs-head 2 --epochs-finetune 2
python scripts/02_train.py --no-finetune

# Evaluation
python scripts/03_evaluate.py
python scripts/03_evaluate.py --threshold 0.42 --split val

# Inference
python -m src.predict path/to/chip.jpg
python -m src.predict path/to/folder --csv results.csv --threshold 0.6

# Scene scanning
python -m src.scan scene.png --metres-per-pixel 10 --overlap 0.5

# Everything
python scripts/run_pipeline.py --quick
```

---

## Tests

```bash
pip install pytest
pytest
```

The suite covers config validation, the synthetic generator, label mapping,
split arithmetic, manifest integrity (including "no image in two splits"), the
tf.data pipeline, model shapes and probability ranges, freeze/unfreeze
behaviour, BatchNorm handling during fine-tuning, `.keras` round-tripping,
Grad-CAM output, robustness to odd input sizes and channel counts, sliding-window
coverage, IoU merging, and a one-epoch end-to-end training run. It needs no
downloaded data and no pretrained weights.

---

## Troubleshooting

**`pip install tensorflow` fails on Windows.** TensorFlow needs 64-bit Python
3.9–3.12. Check with `python -c "import struct; print(struct.calcsize('P')*8)"`
— it must print `64`. Python 3.13 is not supported by TensorFlow yet.

**The EuroSAT download fails.** Corporate networks and university proxies often
block the mirrors. Either download `EuroSAT_RGB.zip` manually in a browser and
drop it in `data/raw/`, then re-run step 1 — or use
`python scripts/01_download_data.py --synthetic` to work entirely offline.

**Out of memory during training.** Set `data.cache: false` and
`data.batch_size: 8` in `config.yaml`, or cap the dataset with
`--max-per-class 500`.

**Training is very slow.** That is normal on CPU. Use `--quick`, reduce
`data.image_size` to `[96, 96]`, or keep `mobilenetv2` (much the fastest of the
three backbones). On an NVIDIA GPU, install `tensorflow[and-cuda]` and set
`train.mixed_precision: true`.

**Streamlit says "no trained model".** Run steps 1 and 2 first, or use the
Pipeline page in the app.

**The app opens but pages error on import.** Launch it from the project root
(`streamlit run app/Home.py`), not from inside `app/`.

**Grad-CAM shows nothing.** An untrained or barely trained model produces flat
gradients. Train for a few real epochs first.

---

## Limitations

Summarised here, argued in full in `reports/METHODOLOGY.md`:

1. The positive class is a **proxy** for waste-site-like surfaces, not verified landfills.
2. Quarries, construction sites, bare ploughed fields and large flat roofs are systematic false alarms.
3. At 10 m resolution, anything under roughly 30 m across is not resolvable.
4. Training data is European; expect degradation in other biomes and seasons.
5. RGB only — SWIR, thermal and methane bands would substantially improve real discrimination.
6. Scene-scan coordinates are pixel offsets, not latitude/longitude.
7. Single-date inference; landfills are best identified by *change over time*.
8. The test split shares a distribution with training; real deployment needs a geographically held-out region.

---

## Credits

- **EuroSAT** — P. Helber, B. Bischke, A. Dengel, D. Borth, *EuroSAT: A Novel
  Dataset and Deep Learning Benchmark for Land Use and Land Cover
  Classification*, IEEE JSTARS, 2019.
- **Grad-CAM** — R. R. Selvaraju et al., *Grad-CAM: Visual Explanations from
  Deep Networks via Gradient-based Localization*, ICCV 2017.
- **MobileNetV2** — M. Sandler et al., CVPR 2018.
- Imagery: Copernicus Sentinel-2, European Space Agency.
