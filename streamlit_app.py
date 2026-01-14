from pathlib import Path
from clinical_early_warning_e2e import run_ui

BASE = Path(__file__).parent

run_ui(
    prepared_npz=BASE / "data" / "processed" / "prepared_dataset.npz",
    model_path=BASE / "artifacts" / "best_model.pt",
    threshold=0.6,
    cases_dir=BASE / "cases"
)
