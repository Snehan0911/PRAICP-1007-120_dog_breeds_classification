"""
CanineVision AI — Centralised Path Management.

Why this matters:
    Hard-coded paths like "d:/Projects/..." break the moment someone else
    clones the project or you move a folder. Relative paths work from any
    machine as long as you run scripts from the project root.

    This module computes all project paths relative to the actual project
    root directory (where this file ultimately lives), so the application
    works regardless of where the project is cloned or what OS is used.

Usage:
    from src.utils.paths import ProjectPaths
    paths = ProjectPaths()
    print(paths.dataset_root)   # → Path object to the dataset folder
    print(paths.artifacts_models)  # → Path object to saved models folder
"""

from pathlib import Path


class ProjectPaths:
    """
    Centralised, auto-discovered project path manager.

    All paths are computed relative to the project root, which is
    determined by locating the 'configs' directory — a reliable marker
    of the project root regardless of the current working directory.

    Attributes:
        root:                  Project root directory.
        dataset_root:          Path to the raw dataset (120 breed folders).
        data_processed:        Path to processed metadata and CSVs.
        data_splits:           Path to train/val/test CSV split files.
        artifacts_models:      Path to saved .keras model files.
        artifacts_class_mapping: Path to class_indices.json.
        artifacts_metadata:    Path to model metadata JSON files.
        artifacts_reports:     Path to evaluation report files.
        outputs_eda:           Path to EDA charts and reports.
        outputs_training:      Path to training history plots and CSVs.
        outputs_evaluation:    Path to evaluation charts.
        outputs_gradcam:       Path to Grad-CAM visualizations.
        outputs_predictions:   Path to prediction history JSON.
        configs:               Path to the configs directory.
        mlflow_tracking:       Path to local MLflow tracking directory.
    """

    def __init__(self, project_root: str | Path | None = None) -> None:
        """
        Initialise project paths.

        Args:
            project_root: Explicit project root. If None, auto-detects
                          by walking up from this file's location until
                          the 'configs' directory is found.
        """
        if project_root is not None:
            self.root = Path(project_root).resolve()
        else:
            self.root = self._find_project_root()

        # --- Raw dataset ---
        self.dataset_root = self.root / "dog-breed-120" / "Dogs class"

        # --- Processed data ---
        self.data_processed = self.root / "data" / "processed"
        self.data_splits = self.root / "data" / "splits"

        # --- Artifacts ---
        self.artifacts_models = self.root / "artifacts" / "models"
        self.artifacts_class_mapping = self.root / "artifacts" / "class_mapping"
        self.artifacts_metadata = self.root / "artifacts" / "metadata"
        self.artifacts_reports = self.root / "artifacts" / "reports"

        # --- Outputs ---
        self.outputs_eda = self.root / "outputs" / "eda"
        self.outputs_training = self.root / "outputs" / "training"
        self.outputs_evaluation = self.root / "outputs" / "evaluation"
        self.outputs_gradcam = self.root / "outputs" / "gradcam"
        self.outputs_predictions = self.root / "outputs" / "predictions"

        # --- Configs ---
        self.configs = self.root / "configs"

        # --- MLflow ---
        self.mlflow_tracking = self.root / "mlruns"

    def _find_project_root(self) -> Path:
        """
        Walk up the directory tree from this file until the 'configs'
        directory is found. This makes the path system work correctly
        whether you run scripts from the project root, src/, or anywhere else.

        Returns:
            Resolved absolute Path to the project root.

        Raises:
            RuntimeError: If the project root cannot be found.
        """
        # Start from the directory containing THIS file (src/utils/)
        current = Path(__file__).resolve().parent

        # Walk up a maximum of 6 levels to avoid infinite loops
        for _ in range(6):
            if (current / "configs").exists() and (current / "src").exists():
                return current
            current = current.parent

        raise RuntimeError(
            "Could not locate the project root. "
            "Ensure 'configs/' and 'src/' directories exist. "
            "Run scripts from the project root directory."
        )

    def ensure_all_exist(self) -> None:
        """
        Create all output and artifact directories if they do not already exist.
        This is called once at application startup to avoid FileNotFoundError
        later when writing outputs.
        """
        dirs_to_create = [
            self.data_processed,
            self.data_splits,
            self.artifacts_models,
            self.artifacts_class_mapping,
            self.artifacts_metadata,
            self.artifacts_reports,
            self.outputs_eda,
            self.outputs_training,
            self.outputs_evaluation,
            self.outputs_gradcam,
            self.outputs_predictions,
            self.mlflow_tracking,
        ]
        for directory in dirs_to_create:
            directory.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return f"ProjectPaths(root={self.root})"
