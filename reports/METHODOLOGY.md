# Methodology and honest limitations

*Landfill Detection from Satellite Imagery — technical report*

---

## 1. Problem framing

Unmonitored and illegal waste sites are hard to find from the ground and easy to
see from orbit — if someone looks. Copernicus Sentinel-2 images the entire
landmass of the planet at 10 m true-colour resolution every five days and gives
it away for free. That is vastly more imagery than any environmental agency can
review by eye.

The job of this system is therefore **triage, not adjudication**: score every
tile in a scene, and hand a human analyst a short ranked list of candidates with
an explanation attached to each one. A false alarm costs a few minutes of an
analyst's time. A missed site costs an unmonitored leachate plume. The system is
tuned and evaluated with that asymmetry in mind.

---

## 2. Data

**EuroSAT** (Helber, Bischke, Dengel & Borth, IEEE JSTARS 2019)

| property | value |
|---|---|
| images | 27,000 |
| chip size | 64 × 64 px |
| ground resolution | 10 m/px → 640 m per chip |
| bands used here | RGB (B4, B3, B2) |
| classes | 10 land-cover types |
| coverage | 34 European countries |
| licence | freely redistributable, research use |

Class counts: `AnnualCrop` 3000, `Forest` 3000, `HerbaceousVegetation` 3000,
`Highway` 2500, `Industrial` 2500, `Pasture` 2000, `PermanentCrop` 2500,
`Residential` 3000, `River` 2500, `SeaLake` 3000.

It was chosen because it is the standard land-use benchmark, small enough to
train on a laptop CPU, and legally redistributable — which matters for a project
meant to be reproducible by anyone who clones it.

---

## 3. The label problem, stated plainly

**EuroSAT has no landfill class.** No freely available, permissively licensed,
globally distributed landfill segmentation dataset exists at the time of
writing. There were three honest options:

1. **Fabricate labels** by hand-picking chips that "look like" landfills.
   Rejected: the labels would encode the author's guesses, and the reported
   accuracy would be a measurement of self-consistency, not of detection.
2. **Ship an untrained pipeline** and tell the user to find their own data.
   Rejected: nothing would be demonstrable or verifiable out of the box.
3. **Train a clearly labelled proxy task**, document the gap loudly, and make
   swapping in real labels a two-line config change. **Chosen.**

### The proxy

| role | land covers |
|---|---|
| positive — *waste-site-like surface* | `Industrial`, `Highway` |
| negative — *clean land* | `Forest`, `Pasture`, `HerbaceousVegetation`, `River`, `SeaLake`, `AnnualCrop`, `PermanentCrop`, `Residential` |

**Why this is defensible.** An active landfill in 10 m true colour is a large,
bare, disturbed, high-albedo anthropogenic surface, criss-crossed by haul roads
and terraced into working faces. That description matches `Industrial`
hardstanding and `Highway` corridors far more closely than it matches any
natural class. A model that separates this positive set from the negative set
has learned "disturbed bare anthropogenic surface versus natural cover" — which
is genuinely most of the landfill-detection signal available in three visible
bands.

**Why `Residential` sits on the negative side.** Including it as a positive
would collapse the task into "built-up versus not built-up", which any
classifier solves trivially and which is useless for finding waste sites inside
developed areas. Forcing housing into the negative class makes the model
discriminate *bare disturbed industrial* from *ordinary built-up*, which is the
harder and more relevant boundary.

**What this does not give you.** It does not give you a validated landfill
classifier. A reported 97% accuracy means "97% of the time it correctly told
industrial/road surfaces apart from natural cover". It does **not** mean 97% of
landfills are found. Section 8 lists every consequence of that gap.

### Swapping in real labels

```
data/custom/landfill/*.jpg
data/custom/not_landfill/*.jpg
```

```yaml
data:
  source: custom
task:
  positive_classes: ["landfill"]
  negative_classes: ["not_landfill"]
```

Re-run steps 1 and 2. The manifest builder, tf.data pipeline, training loop,
evaluation, Grad-CAM, scene scanner and the whole web console adapt
automatically — no code changes.

---

## 4. Preprocessing

### Manifest-based splitting

Copying 27,000 files into `train/`, `val/` and `test/` folders is slow, triples
disk usage, and makes the split opaque. Instead the pipeline writes a single
**manifest** — `data/interim/manifest.csv` — with one row per image:

```
path, source_class, label, label_idx, split
```

Consequences worth having:

- **Reproducible.** Seeded RNG; the same seed always yields the same split.
- **Auditable.** The exact composition of every split is one `read_csv` away.
- **Cheap to regenerate.** Change the task definition, rebuild in seconds.
- **Stratified on the source class**, not just the binary label — so `train`,
  `val` and `test` each keep the same land-cover mix, and a metric can never be
  flattered by, say, all the `SeaLake` chips landing in the test split.
- **Small classes protected.** Any class with ≥ 3 images is guaranteed at least
  one validation and one test image.

Split fractions: 70 / 15 / 15.

### The tf.data pipeline

```
paths → parallel decode → resize 128×128 → cache → shuffle → batch → augment → prefetch
```

Pixels stay in `[0, 255]` throughout; backbone-specific normalisation lives
*inside* the model. That means the exported `.keras` file is self-contained and
there is no way for training and serving preprocessing to drift apart — a
common and silent source of production degradation.

Caching after decode is the single biggest speed win (roughly 3–4× on EuroSAT);
`data.cache: false` turns it off for memory-constrained machines.

### Augmentation

Training split only: horizontal **and** vertical flips, ±25% rotation, 15% zoom,
10% translation, 15% brightness and contrast jitter.

Vertical flips and full rotations are legitimate here in a way they would not be
for ordinary photographs: a nadir satellite view has no canonical "up". Refusing
to exploit that symmetry would throw away free data. Brightness and contrast
jitter stand in for atmospheric and seasonal illumination variation.

---

## 5. Model

```
image (0–255) → Rescaling → backbone → head → P(landfill candidate)
```

**Backbone.** MobileNetV2 by default — 2.3 M parameters, depthwise-separable
convolutions, comfortably the fastest of the three options on CPU.
EfficientNetB0 and ResNet50V2 are drop-in alternatives via config.

**Head.** Global average pooling → dropout 0.3 → dense 256 + ReLU (L2 1e-4) →
batch norm → dropout 0.15 → sigmoid.

**Architectural choice worth noting.** `backbone` and `head` are kept as *named
nested Keras models* rather than being flattened into one layer graph. This
makes Grad-CAM trivial and version-proof: run the backbone, watch its output
tensor, run the head, differentiate. No reaching into layer graphs by name, no
breakage when Keras changes its internals.

**Resolution.** EuroSAT chips are 64 × 64; the network sees 128 × 128. Upsampling
costs nothing in information but matches the receptive-field scale the ImageNet
backbones were trained at, which measurably helps transfer.

---

## 6. Training

Two phases, for a specific reason: a randomly initialised head attached to a
pretrained backbone produces enormous early gradients, and back-propagating
those into carefully learned ImageNet filters destroys them within a few dozen
steps.

**Phase 1 — head warm-up.** Backbone frozen. lr 1e-3, Adam. 8 epochs. Only the
new head learns; the pretrained representation is untouched.

**Phase 2 — fine-tuning.** Deepest 40 backbone layers unfrozen. lr 2e-5 — fifty
times lower. 12 epochs. Early layers stay frozen because edge and texture
detectors transfer perfectly well from natural images; only the high-level,
semantically specific layers need to adapt to Sentinel-2 statistics.

**BatchNorm stays frozen throughout phase 2.** Updating running statistics on a
small dataset whose distribution differs from ImageNet's is a classic and
well-documented cause of fine-tuning collapse — validation accuracy falls off a
cliff while training accuracy keeps climbing.

**Also on by default:** inverse-frequency class weights (the proxy task is
roughly 1:3 positive:negative), early stopping on validation AUC with best-weight
restore, `ReduceLROnPlateau` (factor 0.4, patience 2), NaN termination, and a
per-epoch CSV log.

**Why AUC and not accuracy for early stopping.** Accuracy depends on an
arbitrary 0.5 threshold and is easy to game under class imbalance. AUC measures
ranking quality independently of where the threshold ends up — which matters
because the threshold is chosen *after* training, from the sweep in the
evaluation report.

---

## 7. Evaluation

Written to `reports/`:

| artifact | contents |
|---|---|
| `metrics.json` | every headline number, machine readable |
| `predictions_test.csv` | per-image scores for further analysis |
| `threshold_sweep.csv` | precision/recall/F1/specificity at 101 thresholds |
| `figures/confusion_matrix.png` | counts and row-normalised recall |
| `figures/roc_pr_curves.png` | ROC and precision-recall with baselines |
| `figures/threshold_sweep.png` | operating-point selection |
| `figures/per_source_class.png` | model response by land cover |
| `figures/error_gallery.png` | the most confident mistakes |

**Metrics reported:** accuracy, balanced accuracy, macro and weighted F1,
Matthews correlation, Cohen's κ, ROC AUC, average precision, per-class
precision/recall/F1, specificity, false-alarm rate and miss rate.

Balanced accuracy and MCC are given prominence because plain accuracy is
misleading under a 1:3 class ratio.

**The per-land-cover breakdown is the important table.** It reports the mean
positive score for every *original* EuroSAT class, so you can see precisely
which surfaces the model over-fires on. Land covers with a high mean score but a
negative task label are the systematic false alarms — and adding hard negatives
from exactly those classes is the cheapest available improvement to real-world
precision.

---

## 8. Limitations

1. **The positive class is a proxy.** Metrics measure industrial/road versus
   natural cover, not landfill versus everything else. Do not quote the headline
   accuracy as a landfill-detection rate.

2. **Confusable surfaces.** Quarries, gravel pits, construction sites, bare
   ploughed fields and large flat industrial roofs share the spectral and
   textural signature of an active waste cell in three visible bands. The
   per-land-cover table quantifies this; read it before trusting any result.

3. **Resolution ceiling.** One EuroSAT chip covers 640 m. Fly-tipping, small
   rural dumps and anything under roughly 30 m across is simply not resolvable
   at 10 m/px, regardless of model quality.

4. **Geographic and seasonal bias.** Training data is European. Arid, tropical
   and high-latitude landscapes have different soil and vegetation spectra;
   expect the false-alarm rate to rise outside Europe and outside the growing
   season.

5. **RGB only.** Landfill discrimination improves substantially with SWIR
   (surface moisture and mineralogy), thermal (decomposition heat) and
   methane-sensitive bands. Three visible bands is the weakest usable input.

6. **No georeferencing by default.** Scene-scan coordinates are pixel offsets.
   Real deployment requires reading GeoTIFF affine transforms (`rasterio`) to
   produce latitude/longitude.

7. **Single-date inference.** Landfills are defined as much by *change over
   time* — growth, cell rotation, cover material moving — as by appearance. A
   time-series model over repeat passes would be considerably stronger.

8. **No independent validation region.** Test chips share a distribution with
   training chips. Honest deployment estimates require a geographically
   held-out region, ideally a different country.

9. **Probabilities are not calibrated.** The sigmoid output ranks well but its
   numeric value should not be read as a true probability without Platt scaling
   or isotonic regression fitted on the validation split.

---

## 9. Where to take it next

Roughly in order of expected value per unit of effort:

1. **Real labels.** Everything above is limited by the proxy. A few hundred
   verified chips would change the project's character entirely.
2. **Multispectral input.** Swap in EuroSAT-MS and widen the first convolution
   to 13 bands. SWIR alone should meaningfully separate bare waste from bare soil.
3. **Change detection.** Stack two dates as a 6-channel input. Growth is the
   single strongest signal that a bare patch is an *active* landfill.
4. **Segmentation.** Replace the classifier head with a U-Net decoder for
   pixel-level site outlines instead of boxes.
5. **Georeferencing.** Read GeoTIFF transforms, emit GeoJSON, drop straight
   into QGIS.
6. **Calibration.** Platt scaling on the validation split, so reported
   probabilities can be used as probabilities.
7. **Hard-negative mining.** Feed the model's own top false positives back in as
   negatives and retrain — usually the fastest precision gain available.

---

## References

- Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019). *EuroSAT: A Novel
  Dataset and Deep Learning Benchmark for Land Use and Land Cover
  Classification.* IEEE JSTARS, 12(7), 2217–2226.
- Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D.
  (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based
  Localization.* ICCV.
- Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018).
  *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR.
- Tan, M., & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for
  Convolutional Neural Networks.* ICML.
- Copernicus Sentinel-2 imagery, European Space Agency.
