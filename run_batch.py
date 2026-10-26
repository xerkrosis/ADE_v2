# Executable batch entry point for the ADE pipeline
"""Internal ADE v2 application candidate. No development runners are needed."""

import argparse
from contextlib import redirect_stdout
import importlib
import os
from pathlib import Path
import sys

try:
    import winsound
except ImportError:  # Allows non-Windows import and syntax checks.
    winsound = None

from ade import config
from ade.thresholds import order_parameter_sets, validate_threshold_evaluation_mode


# Dataset, Data file, Data file ending. Empty list: all files in Data/DOSE-I.
datasets_to_run = [
    ("DOSE-I", "10-006", ".csv"),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", action="append", help="Recording ID; repeatable")
    parser.add_argument("--dataset", default="DOSE-I")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without writing files")
    parser.add_argument("--check", action="store_true", help="Check dependencies and enabled models without raw data")
    parser.add_argument("--no-autoencoder", action="store_true")
    parser.add_argument("--no-rf", action="store_true")
    return parser.parse_args()


def hybrid_config():
    from autoencoder.hybrid_v25 import HybridConfig
    return HybridConfig(
        v2_checkpoint=config.AUTOENCODER_V2_CHECKPOINT,
        v3_checkpoint=config.AUTOENCODER_V3_CHECKPOINT,
        device_name=config.AUTOENCODER_DEVICE,
        batch_size=config.AUTOENCODER_BATCH_SIZE,
    )


def check_environment(methods):
    # Import the real modules: checking a module spec alone misses broken installs.
    for name in ("numpy", "pandas", "scipy", "matplotlib", "openpyxl", "mne",
                 "autoreject", "pywt", "sklearn", "joblib"):
        module = importlib.import_module(name)
        print(f"{name}: {getattr(module, '__version__', 'OK')}", flush=True)
    from ade.pipeline import run_pipeline  # noqa: F401; checks the complete ADE import closure
    from ade.quality import load_ml_quality_model_bundle
    for params in methods:
        if params.get("apply_ml_quality_model", False):
            bundle = load_ml_quality_model_bundle(params["ml_quality_model_path"])
            if (bundle.get("target") != "STRICT_CLEAN_LABEL"
                    or bundle.get("cohort_filter") != "usable_only"):
                raise ValueError("RF model has an incompatible target/cohort.")
            import numpy as np
            import pandas as pd
            columns = bundle["feature_columns"]
            probability = bundle["model"].predict_proba(
                pd.DataFrame(np.zeros((1, len(columns))), columns=columns)
            )
            if not np.isfinite(probability).all():
                raise RuntimeError("RF forward check returned non-finite values.")
            print(f"RF bundle and forward check: PASS ({len(columns)} features)")
            break
    if any(p.get("signal_method") == "autoencoder_v25_hybrid" for p in methods):
        from autoencoder.hybrid_v25 import HybridDenoiser
        import torch
        denoiser = HybridDenoiser(hybrid_config())
        device, models = denoiser.load_models()
        with torch.inference_mode():
            for model, checkpoint, path in models:
                values = torch.zeros((2, 1, 625), device=device)
                markers = torch.zeros(
                    (2, len(checkpoint["model_config"]["marker_names"])), device=device
                )
                output = model(values, reference=values, markers=markers)
                if output.shape != values.shape or not torch.isfinite(output).all():
                    raise RuntimeError(f"Checkpoint forward check failed: {path}")
                print(f"{path.name}: PASS ({sum(p.numel() for p in model.parameters()):,} parameters)")
    print("Environment/model check: PASS. No recording was processed.")


def resolve_recordings(args):
    if args.recording:
        recordings = []
        for filename in args.recording:
            suffix = next((ending for ending in (".csv", ".csv.gz")
                           if (config.base_path / args.dataset / (filename + ending)).is_file()), None)
            if suffix is None:
                raise FileNotFoundError(f"No .csv/.csv.gz found for {args.dataset}/{filename}.")
            recordings.append((args.dataset, filename, suffix))
        return recordings
    if datasets_to_run:
        return list(datasets_to_run)
    # Exclusion list by filename
    exclude_files = {
        # "10-123",
    }
    from ade.data_io import discover_datasets_to_run
    return discover_datasets_to_run(config.base_path, args.dataset, exclude_files)


def main():
    args = parse_args()
    threshold_evaluation_mode = validate_threshold_evaluation_mode(config.THRESHOLD_EVALUATION_MODE)
    methods = []
    for parameters in config.parameter_sets:
        params = dict(config.common_params)
        params.update(parameters)
        if args.no_autoencoder and str(params.get("signal_method")).startswith("autoencoder_"):
            continue
        if args.no_rf:
            params["apply_ml_quality_model"] = False
        methods.append(params)
    needs_hybrid = any(p.get("signal_method") == "autoencoder_v25_hybrid" for p in methods)
    # Hybrid generation always needs a same-recording original reference first.
    configured_parameter_sets = order_parameter_sets(
        methods, "fixed_original_thresholds" if needs_hybrid else threshold_evaluation_mode
    )
    if not configured_parameter_sets:
        raise ValueError("No parameter sets remain selected.")
    if args.check:
        check_environment(configured_parameter_sets)
        return 0
    resolved_datasets_to_run = resolve_recordings(args)
    comparison_file_path = config.file_path_sheet
    if threshold_evaluation_mode == "fixed_original_thresholds":
        comparison_file_path = comparison_file_path.with_name(
            f"{comparison_file_path.stem}_fixedthr{comparison_file_path.suffix}"
        )
    print("Python:", sys.executable)
    print("Batch threshold evaluation mode:", threshold_evaluation_mode)
    print("Comparison workbook:", comparison_file_path)
    print("Recordings:", len(resolved_datasets_to_run))
    for dataset, filename, ending in resolved_datasets_to_run:
        if not (config.base_path / dataset / (filename + ending)).is_file():
            raise FileNotFoundError(config.base_path / dataset / (filename + ending))
        print(f"  {dataset}/{filename}: " + ", ".join(p.get("signal_method", "original") for p in configured_parameter_sets))
    if args.dry_run:
        print("Dry run completed. No models loaded and no files written.")
        return 0

    from ade.pipeline import run_pipeline
    denoiser = None
    if needs_hybrid:
        from autoencoder.hybrid_v25 import HybridDenoiser
        denoiser = HybridDenoiser(hybrid_config())
    failed_runs = []                                        # Failsafe, if there are failed runs
    total_runs = len(resolved_datasets_to_run) * len(configured_parameter_sets)
    current_run = 0                                         # Progress Display: Starting Parameters
    comparison_file_path.parent.mkdir(parents=True, exist_ok=True)

    for dataset, filename, file_ending in resolved_datasets_to_run:
        original_threshold_reference = None
        original_params = None
        for method_params in configured_parameter_sets:
            current_run += 1
            params_run = dict(config.common_params)
            params_run.update(method_params)
            params_run["threshold_evaluation_mode"] = threshold_evaluation_mode
            method_name = params_run.get("signal_method", "original")
            try:
                if method_name == "original":
                    if needs_hybrid or threshold_evaluation_mode == "fixed_original_thresholds":
                        params_run["export_artifact_windows"] = True
                        params_run["_capture_original_thresholds"] = True
                    if needs_hybrid:
                        params_run["export_autoencoder_windows"] = True
                elif threshold_evaluation_mode == "fixed_original_thresholds":
                    if original_threshold_reference is None:
                        raise RuntimeError(f"No successful original threshold reference for {dataset}/{filename}.")
                    params_run["_original_threshold_reference"] = original_threshold_reference

                ml_threshold = float(params_run.get("ml_strict_clean_threshold", config.ML_STRICT_CLEAN_THRESHOLD))
                if not 0.0 <= ml_threshold <= 1.0:
                    raise ValueError("ml_strict_clean_threshold must be between 0.0 and 1.0.")
                params_run["ml_strict_clean_threshold"] = ml_threshold
                if params_run.get("apply_ml_quality_model", False):
                    model_path = Path(params_run["ml_quality_model_path"]).expanduser().resolve()
                    if not model_path.is_file():
                        raise FileNotFoundError(f"ML quality model not found: {model_path}")
                    params_run["ml_quality_model_path"] = model_path
                    print("ML strict-clean threshold:", ml_threshold)

                artifact_variant = config.build_artifact_variant(params_run)
                params_run["artifact_export_dir"] = config.artifact_export_root / artifact_variant
                if params_run.get("export_autoencoder_windows", False):
                    source_root = Path(params_run["autoencoder_export_dir"])
                    params_run["autoencoder_export_dir"] = source_root / dataset / artifact_variant

                if method_name == "autoencoder_v25_hybrid":
                    if original_params is None or original_threshold_reference is None:
                        raise RuntimeError("The original run failed; hybrid inference cannot continue.")
                    prediction_dir = Path(params_run["autoencoder_prediction_dir"]) / dataset
                    prediction_path = prediction_dir / f"{filename}_windows_denoised.npz"
                    work_dir = config.AUTOENCODER_WORK_ROOT / dataset / filename

                    def assess_v21(candidate_dir, artifact_dir):
                        candidate_params = dict(original_params)
                        candidate_params.update({
                            "signal_method": "autoencoder_v21",
                            "autoencoder_prediction_dir": candidate_dir,
                            "autoencoder_expected_policy_version": "v2.1",
                            "autoencoder_expected_ecg_min_abs_correlation": 0.05,
                            "autoencoder_expected_bypass_markers": ("FLATLINE", "CLIPPING"),
                            "artifact_export_dir": artifact_dir,
                            "export_artifact_windows": True,
                            "export_autoencoder_windows": False,
                            "apply_ml_quality_model": False,
                            "make_plots": False,
                            "threshold_evaluation_mode": "fixed_original_thresholds",
                            "_original_threshold_reference": original_threshold_reference,
                        })
                        work_dir.mkdir(parents=True, exist_ok=True)
                        log_path = work_dir / "candidate_ade.log"
                        print("Assessing v2 candidate with current original thresholds...", flush=True)
                        try:
                            with log_path.open("w", encoding="utf-8") as log, redirect_stdout(log):
                                run_pipeline(dataset=dataset, filename=filename, file_ending=file_ending,
                                             params=candidate_params, base_path=config.base_path,
                                             file_path_sheet=work_dir / "candidate_metrics.xlsx")
                        except Exception:
                            print("Candidate assessment log:", log_path)
                            raise
                        return artifact_dir / f"{filename}_artifacts.xlsx"

                    denoiser.apply_recording(
                        recording_id=filename,
                        input_path=Path(original_params["autoencoder_export_dir"]) / f"{filename}_windows.npz",
                        original_artifact_path=Path(original_params["artifact_export_dir"]) / f"{filename}_artifacts.xlsx",
                        work_dir=work_dir, output_path=prediction_path, assess_v21=assess_v21,
                    )
                    params_run["autoencoder_prediction_dir"] = prediction_dir
                elif str(method_name).startswith("autoencoder_"):
                    prediction_path = Path(params_run["autoencoder_prediction_dir"]) / f"{filename}_windows_denoised.npz"
                    if not prediction_path.is_file():
                        raise FileNotFoundError(f"Autoencoder prediction not found: {prediction_path}")

                if os.name == "nt":
                    os.system(f"title EEG Batch [{current_run}/{total_runs}] {filename} {method_name}")
                print("=" * 80)
                print(f"[{current_run}/{total_runs}] Starting: {dataset} / {filename} / {method_name}")
                print("Artifact export directory:", params_run["artifact_export_dir"])
                print("=" * 80)
                # Start Pipeline
                pipeline_result = run_pipeline(
                    dataset=dataset, filename=filename, file_ending=file_ending,
                    params=params_run, base_path=config.base_path,
                    file_path_sheet=comparison_file_path,
                )
                if method_name == "original" and (needs_hybrid or threshold_evaluation_mode == "fixed_original_thresholds"):
                    if not isinstance(pipeline_result, dict):
                        raise RuntimeError("The pipeline did not return original thresholds.")
                    original_threshold_reference = pipeline_result
                    original_params = params_run.copy()
            except Exception as error:
                failed_runs.append({"dataset": dataset, "filename": filename,
                                    "signal_method": method_name, "error": str(error)})
                print("[ERROR]", failed_runs[-1], flush=True)

    print("Batch completed.")
    if os.name == "nt":
        os.system("title EEG Batch completed")
    if failed_runs:
        print("Failed runs:", failed_runs)
        if winsound is not None:
            winsound.MessageBeep(winsound.MB_ICONHAND)      # Error SFX
        return 1
    print("No errors.")
    if winsound is not None:
        winsound.MessageBeep(winsound.MB_ICONASTERISK)      # Done SFX
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
