# ADE v2 - An Artifact Detection Framework for Two-Channel EEG

A configurable Python framework for preprocessing two-channel EEG recordings, detecting potential artifacts, assessing signal quality, and comparing conventional and learned denoising methods.

ADE v2 extends [ADE v1](https://github.com/safe-ai-research/ADE) with Random-Forest quality scores, a pretrained autoencoder hybrid, and comparisons that reuse the original signal's detection thresholds. It was developed by Andreas Schwab in the context of the SAFE-AI research project.

The project uses the public DOSE-I dataset. Raw recordings and the development experiment results are not included in this repository.

[![DOSE-I DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18483292.svg)](https://doi.org/10.5281/zenodo.18483292)

> This is research software. Artifact markers and quality labels are heuristic indicators, and Random-Forest outputs are uncalibrated model scores. They are not clinical diagnoses or expert ground truth.

The current application version is recorded in [`VERSION`](VERSION) as `2.0.0-rc1-internal`. This repository contains the application and supplied models, RF training and evaluation tools, and the autoencoder inference implementation. Neural-network training and the full development experiment suite are not included.

## Contents

- [Processing overview](#processing-overview)
- [Project structure](#project-structure)
- [Expected input format](#expected-input-format)
- [Requirements and installation](#requirements-and-installation)
- [Running the pipeline](#running-the-pipeline)
- [Configuration](#configuration)
- [Features, markers, and rule-based labels](#features-markers-and-rule-based-labels)
- [Random-Forest quality assessment](#random-forest-quality-assessment)
- [Autoencoder v2.5 hybrid](#autoencoder-v25-hybrid)
- [Output files](#output-files)
- [Training and evaluating your own RF model](#training-and-evaluating-your-own-rf-model)
- [Limitations and failure behaviour](#limitations-and-failure-behaviour)
- [Data handling and research use](#data-handling-and-research-use)
- [Related project and attribution](#related-project-and-attribution)
- [License](#license)

## Features

- CSV and compressed CSV (`.csv.gz`) input, with individual recording selection or batch discovery.
- Resampling onto a uniform 125 Hz time axis.
- Configurable detrending, DC removal, high-pass, low-pass, notch filtering, and z-score normalization.
- Conventional processing methods: original preprocessed EEG, ECG-referenced adaptive filtering, wavelet denoising, ICA, AJDC, and experimental ASR.
- Window-based time, frequency, wavelet, and cross-channel features.
- Channel-specific and combined artifact markers and quality labels.
- Optional Random-Forest scores for strict-clean versus usable-but-not-strict windows.
- Autoencoder v2.5 hybrid inference using the supplied v2 and v3 checkpoints.
- Recording-adaptive assessment or comparisons using fixed thresholds from the same recording's original signal.
- Excel exports for recording metrics, all-window features, artifact windows, effective thresholds, and ML application metadata.
- NumPy exports of source and denoised windows, with channel-wise inference decisions.
- Dependency/model checks, dry runs, and optional diagnostic plots.

## Processing Overview

For each selected recording, [`run_batch.py`](run_batch.py) merges the common settings with each configured method and coordinates the following steps:

1. Load timestamps, two EEG channels, and available ECG/heart-rate references.
2. Resample and apply the configured basic preprocessing.
3. Calculate baseline features and create signal windows.
4. Apply the selected signal-processing method.
5. Calculate features, detect artifact markers, and assign rule-based quality labels.
6. Optionally score rule-based usable windows with the RF model.
7. Export window annotations, effective thresholds, and recording-level comparisons.

With the default fixed-threshold mode, the original signal runs first and supplies full-precision thresholds for the other methods on that recording. When the hybrid is selected, the runner also exports its source windows, generates both neural candidates, assesses the v2 candidate through ADE, builds the hybrid, and evaluates the reconstructed signal.

Existing experiment exports are not required for a new hybrid run. Candidate predictions are generated again for the current input and configuration.

## Project Structure

| Path | Responsibility |
|---|---|
| [`run_batch.py`](run_batch.py) | Command-line entry point, recording selection, batch execution, and hybrid orchestration |
| [`ade/config.py`](ade/config.py) | Paths, common parameters, selected methods, model paths, and export naming |
| [`ade/pipeline.py`](ade/pipeline.py) | Processing and assessment of one recording/method |
| [`ade/data_io.py`](ade/data_io.py) | Recording discovery/loading, Excel and source-window exports, and plots |
| [`ade/signal_processing.py`](ade/signal_processing.py) | Resampling, basic preprocessing, conventional methods, and prediction integration |
| [`ade/windowing.py`](ade/windowing.py) | Signal windows, sample positions, and absolute timestamps |
| [`ade/features.py`](ade/features.py) | Recording/window features and engineered channel differences |
| [`ade/artifacts.py`](ade/artifacts.py) | AutoReject, heuristic markers, and hard-artifact rules |
| [`ade/quality.py`](ade/quality.py) | Rule-based labels, RF inference, and application-context reporting |
| [`ade/thresholds.py`](ade/thresholds.py) | Same-recording original-threshold references and compatibility checks |
| [`autoencoder/models.py`](autoencoder/models.py) | Single-channel convolutional denoiser architecture |
| [`autoencoder/inference.py`](autoencoder/inference.py) | Conditioned inference and preservation/fallback policies |
| [`autoencoder/hybrid_v25.py`](autoencoder/hybrid_v25.py) | Candidate generation and selective v2/v3 hybrid decisions |
| [`autoencoder/pipeline_integration.py`](autoencoder/pipeline_integration.py) | Validate prediction provenance and reconstruct continuous EEG |
| [`autoencoder/io.py`](autoencoder/io.py) | NumPy window export and loading |
| [`ml/`](ml/) | Recording/window splits, RF training, evaluation, and model-bundle utilities |
| [`models/`](models/) | Supplied neural checkpoints and archived RF bundle |
| [`requirements.txt`](requirements.txt) | Pinned core and RF dependencies |
| [`requirements-autoencoder.txt`](requirements-autoencoder.txt) | Core dependencies plus PyTorch |
| [`requirements-classic-optional.txt`](requirements-classic-optional.txt) | Optional ASR and AJDC dependencies |

Paths are resolved relative to the project root. Input recordings belong in `Data/<dataset>/`; generated files are written below `results/` by default. These directories are created or populated locally.

## Expected Input Format

Input files must be comma-separated `.csv` or `.csv.gz` files with these columns:

| Column | Required | Purpose |
|---|---|---|
| `Time` | Yes | Timestamps parseable by `pandas.to_datetime` |
| `Intellivue/EEG_1` | Yes | First EEG channel |
| `Intellivue/EEG_2` | Yes | Second EEG channel |
| `Intellivue/ECG_II` or `Intellivue/ECG_II_WAV` | No | ECG waveform reference; the first available candidate is used |
| `Intellivue/ECG_HR` | No | Heart-rate reference values |
| `Intellivue/PLETH_HR` | No | Additional heart-rate reference values |

Rows without a valid timestamp and both numeric EEG values are removed from the EEG input. The ECG waveform is cleaned separately and interpolated onto the EEG time axis. The heart-rate reference is the mean of available numeric values in the heart-rate columns.

Column names and the 125 Hz target rate are currently defined in `load_recording()` in [`ade/data_io.py`](ade/data_io.py). Other export formats require adapting that loader. Signal values are used in their supplied amplitude scale; the loader does not perform an automatic unit conversion.

For the default example, place the recording at:

```text
Data/DOSE-I/10-006.csv
```

## Requirements and Installation

The dependency files capture the working project's **Python 3.12** environment. Versions are pinned; in particular, the supplied RF bundle was saved with **scikit-learn 1.9.1**. Keep that version when loading the supplied bundle.

### Create an environment

Clone the repository and create a virtual environment:

```powershell
git clone https://github.com/xerkrosis/ADE_v2.git
cd ADE_v2
python -m venv .venv
.venv\Scripts\Activate.ps1
```

On Linux/macOS, activate the environment with `source .venv/bin/activate` instead.

### Install dependencies

For the default configuration, including hybrid inference:

```powershell
python -m pip install -r requirements-autoencoder.txt
```

For the conventional pipeline and RF tools without neural inference:

```powershell
python -m pip install -r requirements.txt
```

For the optional ASR and AJDC methods, additionally install:

```powershell
python -m pip install -r requirements-classic-optional.txt
```

The core dependencies include NumPy, pandas, SciPy, Matplotlib, MNE-Python, OpenPyXL, AutoReject, PyWavelets, scikit-learn, and joblib. PyTorch is only required when neural inference is enabled; ASR and AJDC use `asrpy` and `pyriemann` respectively.

The autoencoder dependency file pins `torch==2.13.0`; its comments record `2.13.0+cu130` as the original training build. For GPU execution, use a compatible CUDA-enabled PyTorch build. See the official [PyTorch installation instructions](https://pytorch.org/get-started/locally/) for platform-specific installation. `AUTOENCODER_DEVICE = "auto"` selects CUDA when available and otherwise uses the CPU.

### Prepare the supplied RF model

The RF bundle is distributed as [`models/quality_rf.zip`](models/quality_rf.zip), while the default configuration expects an unpacked `models/quality_rf.joblib`.

Extract the archive before enabling RF inference:

```powershell
python -m zipfile -e models/quality_rf.zip models
```

Check the extracted location. If its filename or directory differs, update `ML_QUALITY_MODEL_PATH` in [`ade/config.py`](ade/config.py) to point to the extracted `.joblib` file. The ZIP archive itself cannot be passed to the model loader.

The hybrid uses both supplied checkpoint files directly:

```text
models/denoiser_v2.pt
models/denoiser_v3.pt
```

### Check the installation

Check dependencies and enabled models without accessing raw recordings:

```powershell
python run_batch.py --check
```

This imports the scientific runtime, loads the enabled models, and checks their outputs on dummy inputs. It does not establish scientific denoising performance.

For an installation without neural inference:

```powershell
python run_batch.py --check --no-autoencoder
```

To check only the conventional runtime, also add `--no-rf`. The checker still imports all packages listed in `requirements.txt`; optional ASR/AJDC packages are not covered by this check.

## Running the Pipeline

All commands below are executed from the repository root.

```powershell
python run_batch.py --help
python run_batch.py --recording 10-006 --dry-run
python run_batch.py --recording 10-006
```

`--dry-run` checks recording selection and prints the batch plan without loading models or writing results. The selected input files must already exist; their contents are not inspected.

The checked-in configuration selects `DOSE-I/10-006.csv` and runs both `original` and `autoencoder_v25_hybrid`, with RF scoring enabled for both. It uses five-second windows without overlap, fixed original thresholds, annotation export, and no plots.

### Command-line options

| Option | Behaviour |
|---|---|
| `--recording ID` | Select a recording by filename without extension; repeat for several recordings |
| `--dataset NAME` | Dataset folder for `--recording` selection or automatic discovery; default: `DOSE-I` |
| `--dry-run` | Print the resolved plan and check file existence without processing |
| `--check` | Import dependencies and check enabled models without raw recordings |
| `--no-autoencoder` | Remove autoencoder methods from the configured batch |
| `--no-rf` | Disable RF scoring for the selected methods |

For example:

```powershell
python run_batch.py --recording 10-006 --recording 10-020
python run_batch.py --dataset MyDataset --recording recording_01
python run_batch.py --recording 10-006 --no-autoencoder --no-rf
```

For explicit `--recording` selection, the runner searches for `.csv` first and `.csv.gz` second. Without that option it uses `datasets_to_run` in [`run_batch.py`](run_batch.py). `--dataset` does not replace dataset names already stored in that list.

For automatic discovery, set:

```python
datasets_to_run = []
```

Then `python run_batch.py --dataset DOSE-I` processes all supported files in `Data/DOSE-I/`, except recording IDs listed in `exclude_files` inside `resolve_recordings()`.

In Visual Studio, select `run_batch.py` as the startup file and use the same environment. The runner returns exit code `1` if any recording/method run raises an error and reports those runs at the end.

## Configuration

Most settings live in [`ade/config.py`](ade/config.py):

| Setting | Purpose / current default |
|---|---|
| `base_path` | Input root: `Data/` |
| `file_path_sheet` | Comparison workbook: `results/EEG_comparison_data.xlsx` before the threshold-mode suffix |
| `artifact_export_root` | Annotation root: `results/Artifact_annotations_pipeline/` |
| `THRESHOLD_EVALUATION_MODE` | `"fixed_original_thresholds"` |
| `parameter_sets` | Selected methods and method-specific overrides |
| `common_params` | Shared preprocessing, window, marker, and export settings |
| `AUTOENCODER_V2_CHECKPOINT`, `AUTOENCODER_V3_CHECKPOINT` | Neural checkpoint paths |
| `AUTOENCODER_DEVICE` | `"auto"`, `"cpu"`, or `"cuda"`; default: `"auto"` |
| `AUTOENCODER_BATCH_SIZE` | Neural inference batch size; default: `256` |
| `APPLY_ML_QUALITY_MODEL` | RF scoring for the configured original run; default: `True` |
| `APPLY_RF_TO_AUTOENCODER` | Exploratory RF scoring of hybrid-derived features; default: `True` |
| `ML_QUALITY_MODEL_PATH` | Unpacked RF `.joblib` path |
| `ML_STRICT_CLEAN_THRESHOLD` | RF recommendation cutoff; default: `0.80` |

Each method dictionary overrides matching entries in `common_params`. Export paths and original-reference settings are then prepared by the runner.

### Preprocessing and windowing

Current common settings include the following values. To apply them while keeping the other settings, update the existing dictionary in `ade/config.py`:

```python
common_params.update({
    "use_detrend": True,
    "use_dc": True,
    "use_highpass": True,       # 0.5 Hz
    "use_lowpass": False,       # Optional 30 Hz low-pass
    "use_notch": False,         # Optional 50 Hz notch
    "use_normalization": False,
    "window_sec": 5,
    "window_overlap": 0.0,
    "run_autoreject": True,
    "use_robust_artifact_markers": True,
    "use_wavelet_marker_features": True,
    "make_plots": False,
})
```

For conventional methods, use a positive window duration and `0 <= window_overlap < 1`, with a nonzero sample step. **The supplied hybrid requires 125 Hz, five-second windows, and zero overlap**: 625 samples per channel/window. Disable the hybrid before experimenting with other window settings.

To display diagnostic figures, set `common_params["make_plots"] = True`.

### Signal-processing methods

| Identifier | Description |
|---|---|
| `original` | Conventionally preprocessed EEG; also supplies the original reference |
| `adaptive_ecg` | ECG-referenced normalized LMS filtering |
| `wavelet` | Wavelet denoising |
| `ica_ecg` | ICA-based removal of ECG-related components |
| `ajdc` | AJDC source separation and reconstruction |
| `asr` | Experimental Artifact Subspace Reconstruction |
| `autoencoder_v25_hybrid` | Generate and apply the supplied v2.5 hybrid |
| `autoencoder_v21` | Integrate existing compatible prediction files; not an automatic standalone inference runner |

For a conventional comparison, replace `parameter_sets`, for example, with:

```python
parameter_sets = [
    {"signal_method": "original", "export_artifact_windows": True},
    {
        "signal_method": "wavelet",
        "wavelet_name": "db4",
        "wavelet_threshold_scale": 0.1,
        "export_artifact_windows": True,
    },
    {
        "signal_method": "adaptive_ecg",
        "adaptive_ecg_filter_len": 32,
        "adaptive_ecg_mu": 0.05,
        "export_artifact_windows": True,
    },
]
```

This example inherits the common default `apply_ml_quality_model=False`. To retain RF scoring for a method, explicitly enable it in that method's dictionary.

### Threshold evaluation

| Mode | Behaviour |
|---|---|
| `recording_adaptive` | Estimate recording-dependent thresholds separately for each method's output |
| `fixed_original_thresholds` | Estimate thresholds on the original, then reuse those numeric values for the other methods on the same recording |

The fixed mode includes AutoReject and adaptive marker thresholds, plus the quality variance cutoff. It passes the original values in memory rather than recovering rounded values from Excel. It checks recording identity, signal/window dimensions, and preprocessing/marker settings before using the reference. A failed original reference is not replaced with an old export.

Fixed comparisons add `_fixedthr` to comparison workbooks and generated annotation-directory names. An original reference is automatically added when needed. The hybrid also requires an original run for candidate generation, even when final assessment is set to `recording_adaptive`.

## Features, Markers, and Rule-Based Labels

Features include absolute and relative delta/theta/alpha/beta power, 30–45 Hz activity, variance, standard deviation, median absolute deviation, peak-to-peak and maximum amplitude, sample jumps, line length, trend slope, clipping ratio, channel correlation, ECG-related features, wavelet energy/statistics, and engineered channel differences. Recording-level comparisons also include reconstruction correlation, variance ratios, and root-mean-square differences.

### Artifact markers

| Marker | Heuristic interpretation |
|---|---|
| `SLOW_CORR` | Strong slow activity with inter-channel correlation |
| `EMG` | Increased beta activity and variance |
| `ECG` | ECG-like periodic activity in the EEG |
| `BAD_AUTO` | Peak-to-peak amplitude exceeds the AutoReject threshold |
| `AMP_BURST` | Unusually large peak-to-peak amplitude |
| `JUMP` | Abrupt sample-to-sample change |
| `FLATLINE` | Very low variance or peak-to-peak amplitude |
| `HF_EMG` | Increased high-frequency activity and line length |
| `LOW_DRIFT` | Slow dominance with a strong window trend |
| `CLIPPING` | Many values near the local minimum or maximum |
| `WAVELET_HF` | High relative energy in high-frequency wavelet details |
| `WAVELET_TRANSIENT` | Unusually large wavelet detail coefficients |

Most markers are available per channel and combined across the two channels. `SLOW_CORR` is shared. `NO_MARKER` is an internal artifact-type value, not a separate exported marker column. Effective thresholds are written to the annotation workbook.

### Quality labels

| Exported label | Definition |
|---|---|
| `QUALITY_BASE_LABEL` | Absolute inter-channel correlation above `0.3` and channel variance below its effective quality cutoff |
| `QUALITY_LABEL` | Quality-base criteria plus an AutoReject veto |
| `USABLE_LABEL` | Quality criteria pass and no configured hard-artifact marker is present |
| `STRICT_CLEAN_LABEL` | Quality criteria pass and no marker is present |

Labels also have `_EEG1` and `_EEG2` variants. A combined label is `1` only when both channel labels are `1`.

The initial quality variance cutoff is the 75th percentile pooled across both channels' window variances; fixed-threshold comparisons retain the original cutoff. `SLOW_CORR` is not a hard artifact. `LOW_DRIFT` and wavelet markers are also non-hard by default; `low_drift_is_hard_artifact` and `wavelet_markers_are_hard_artifacts` change that behaviour.

`STRICT_CLEAN_LABEL` is therefore a subset of `USABLE_LABEL`. Neither label establishes physiological normality or agreement with an independent expert annotation.

## Random-Forest Quality Assessment

The RF model scores only windows with the combined rule-based `USABLE_LABEL == 1`. Within that cohort, it distinguishes:

- Class `0`: usable but not strict-clean.
- Class `1`: strict-clean.

The model targets pipeline-derived labels. It provides an additional score and recommendation while preserving every rule-based label.

| Output column | Meaning |
|---|---|
| `ML_P_USABLE_NOT_STRICT` | Model score for class `0` |
| `ML_P_STRICT_CLEAN` | Model score for class `1` |
| `ML_STRICT_CLEAN_AT_THRESHOLD` | `1` when the class-1 score reaches `ML_STRICT_CLEAN_THRESHOLD`; otherwise `0` |

Non-usable windows remain unscored (`NaN`). These scores are uncalibrated and should not be read as verified probabilities that a window is artifact-free. The default cutoff `0.80` is adjustable.

The documented RF reference setup is `original_v2`: original preprocessed signals, 125 Hz, five-second windows, and no overlap. Application to hybrid-derived features or other window settings is allowed with a warning, which is retained in `ml_summary`. Matching those reference fields alone does not establish generalization to another dataset or preprocessing setup.

## Autoencoder v2.5 Hybrid

The neural backbone is implemented in [`autoencoder/models.py`](autoencoder/models.py): a one-dimensional convolutional encoder/decoder with residual artifact prediction. Each EEG channel is processed independently by the same model. Conditioned models receive an artifact-marker vector and an ECG waveform reference when the inference policy accepts its coupling to that EEG channel.

**Checkpoint versions and inference-policy versions are separate.** The hybrid uses:

| Component | Role |
|---|---|
| `denoiser_v3.pt` with policy `v2.3` | Base output with preservation and fallback checks |
| `denoiser_v2.pt` with policy `v2.1` | Alternative candidate for selected `LOW_DRIFT` cases |
| Hybrid policy `v2.5` | Channel-wise selection between those outputs; no additional network is trained |

The base policy preserves strict-clean channel windows, bypasses `FLATLINE`/`CLIPPING` and low-information inputs, restricts ECG-reference use, and restores the input when output or signal-preservation checks fail.

The hybrid considers a v2 substitution when the v3 candidate fails its low-drift preservation check. The substitution must resolve the same channel's `LOW_DRIFT` marker without new channel markers or simultaneous event/ECG preservation failures. It must also pass waveform-preservation checks, retain at least 80% of the input's absolute 4–30 Hz power, and avoid introducing a new two-channel `SLOW_CORR` marker relative to the v3 base output.

The default preservation limits are correlation at least `0.90` and relative RMSE at most `0.50`, including a high-pass comparison for low drift. These are exploratory policy settings, not clinically validated acceptance limits.

The integration layer checks the source recording, samples, window alignment, policy metadata, and channel-wise decisions before reconstructing continuous EEG. Policy-preserved and rejected windows, plus the unwindowed tail, retain the current preprocessed input samples.

## Output Files

### Recording-level comparison

With the default fixed-threshold configuration:

```text
results/EEG_comparison_data_fixedthr.xlsx
```

Each recording/method is stored as a column, including global features, quality/marker counts, processing status, method parameters, reconstruction metrics, and ML/inference metadata. Adaptive mode uses `EEG_comparison_data.xlsx` instead.

### Per-recording annotations

```text
results/Artifact_annotations_pipeline/<variant>/<recording>_artifacts.xlsx
```

Default variant directories include:

```text
original_w5_o0_fixedthr
autoencoder_v25_hybrid_w5_o0_fixedthr
```

| Sheet | Contents |
|---|---|
| `artifacts` | Windows containing at least one marker, with timestamps, flags, and labels |
| `windows` | All windows, numeric features, marker flags, quality labels, and any RF scores |
| `thresholds` | Effective detection/quality thresholds and reference metadata |
| `ml_summary` | RF model, cutoff, score summaries, and application-context information; present when RF scoring is applied |

`begin_idx` and `end_idx` are absolute timestamps. Labels/marker flags use uppercase export names; numeric feature names remain lowercase. Read `sheet_name="windows"` explicitly for complete window data, since `artifacts` is the first sheet.

### Autoencoder data and decisions

| Path | Contents |
|---|---|
| `results/autoencoder/input/<dataset>/<original_variant>/<recording>_windows.npz` | Current source EEG windows, optional ECG, labels, and preprocessing metadata |
| `results/autoencoder/predictions/<dataset>/<recording>_windows_denoised.npz` | Hybrid output windows and inference/policy metadata |
| `results/autoencoder/work/<dataset>/<recording>/` | v2/v3 candidates, candidate assessment workbook/log, `hybrid_decisions.csv`, and `hybrid_summary.json` |

The main EEG arrays use shape `(windows, 2, samples)`; default windows contain 625 samples. Prediction files contain windowed data and decision metadata rather than a separate continuous CSV recording. ADE reconstructs the continuous signal internally for assessment.

Repeated runs replace annotation/prediction files and matching comparison columns. Generated names do not encode every setting. Use a separate output root, comparison workbook, or explicit `artifact_variant` for experiments whose results must remain separate.

## Training and Evaluating Your Own RF Model

The [`ml`](ml/) package can train a new RF bundle from current ADE annotation workbooks. It does not require the original development splits.

1. Generate original-signal annotations for multiple recordings, for example with `python run_batch.py --no-autoencoder --no-rf` after setting `datasets_to_run = []`.
2. Check [`ml/ml_config.py`](ml/ml_config.py). By default, both feature and label inputs point to `results/Artifact_annotations_pipeline/original_w5_o0_fixedthr/`.
3. Create a split, train, and evaluate from the repository root:

```powershell
python -m ml.create_split
python -m ml.train_random_forest
python -m ml.evaluate_models --partition validation
```

The defaults use a recording-level split (`files`), 70% training / 30% validation / 0% test, split seed `42`, and a 300-tree RF with `class_weight="balanced"`. Training uses median imputation and only the original usable cohort, targeting `STRICT_CLEAN_LABEL`.

Splitting by recording keeps windows from the same recording together. A `windows` split is available for exploratory comparisons, but can place related windows from one recording in different partitions.

Generated files are written below:

```text
results/rf/splits/
results/rf/models/
results/rf/evaluation/
```

Evaluation exports metrics and confusion matrices. It requires the model's saved split and a nonempty evaluation partition containing both classes. The default has no test partition; configure one before requesting `--partition test`. Reusing these settings creates a new local experiment and does not reproduce the supplied model's original held-out results.

Existing split files are protected from accidental replacement. Change `SPLIT_VERSION` or the split seed for a new split, or set `ML_REUSE_EXISTING_SPLIT=1` to reuse the selected split. Supported environment overrides include `ML_FEATURE_DATA_DIR`, `ML_LABEL_DATA_DIR`, `ML_SPLIT_MODE`, `ML_TRAIN_SIZE`, `ML_SPLIT_RANDOM_STATE`, `ML_N_ESTIMATORS`, and `ML_CLASS_WEIGHT`.

To use a newly trained bundle in the pipeline, set `ML_QUALITY_MODEL_PATH` to its generated `.joblib` file. Neural pretraining, fine-tuning, and augmentation runners are outside the scope of this application snapshot.

## Limitations and Failure Behaviour

- The loader, channel definitions, and models target the current two-channel Intellivue/DOSE-I workflow. Other datasets require format and applicability checks.
- The hybrid is restricted to five-second windows without overlap at 125 Hz. RF scores also have a documented signal/window reference context.
- The windowing implementation drops incomplete tails and currently also omits a final window whose end exactly equals the signal length. Autoencoder reconstruction preserves the unwindowed tail.
- Many thresholds are recording-dependent. Fixed-original comparisons hold them constant within one recording; they do not create a universal threshold across recordings.
- Fewer markers or higher model scores alone do not demonstrate successful artifact removal or preservation of physiological EEG.
- ICA/AJDC are constrained by the two-channel input. ASR remains experimental.
- Failed or unavailable conventional processing may fall back to the original preprocessed signal for feature calculation. Its status is recorded and annotation/window export for that unsuccessful method is skipped. Consult method status fields as well as the batch error list.
- Fixed-original assessment requires a successful original threshold reference. Incompatible or missing neural predictions are rejected rather than accepted as a successful denoising result.
- The supplied models depend on compatible package versions; an environment/model check does not replace evaluation on the intended data.

## Data Handling and Research Use

Keep local recordings, derived research data, and experiment outputs out of version control unless they are explicitly intended and permitted for distribution. Suggested local `.gitignore` entries include:

```gitignore
Data/
results/
.venv/
__pycache__/
*.py[cod]
```

This software is intended for research, development, and education. It is not a medical device and must not be used as the sole basis for clinical decisions.

## Related Project and Attribution

- [ADE v1](https://github.com/safe-ai-research/ADE): predecessor framework and the basis for the conventional preprocessing and heuristic assessment workflow.
- [DOSE-I](https://doi.org/10.5281/zenodo.18483292): dataset used in this project; consult its record for data access and citation information.
- The runtime dependencies are listed in the repository's requirements files. Their respective licenses and citation guidance apply to those projects.

## License

ADE v2 is licensed under the [Apache License 2.0](LICENSE).

## Acknowledgements

This work was supported by Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Project number 547230187.
