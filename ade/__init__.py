"""Artifact Detection and Evaluation."""

def run_pipeline(*args, **kwargs):
    # Keep configuration importable without loading the scientific runtime.
    from .pipeline import run_pipeline as implementation
    return implementation(*args, **kwargs)

__all__ = ["run_pipeline"]
