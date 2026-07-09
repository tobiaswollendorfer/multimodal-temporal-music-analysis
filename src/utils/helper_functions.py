from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def load_dataset_csv(dataset_name: str) -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / dataset_name
    return pd.read_csv(path,sep="\t")


def save_dataset_parquet(
    dataset: pd.DataFrame,
    save_name: str,
    save_dir: str | Path | None = None
) -> None:
    target_dir = Path(save_dir) if save_dir is not None else PROCESSED_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    path = target_dir / f"{save_name}.parquet"
    dataset.to_parquet(path, index=False)
    

def load_dataset_parquet(dataset_name: str) -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / dataset_name 
    return pd.read_parquet(path)