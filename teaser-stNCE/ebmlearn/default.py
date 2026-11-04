from pathlib import Path

# Create folders
ROOT_FOLDER = Path(__file__).parent.parent

RESULTS_FOLDER = ROOT_FOLDER / "results"
RESULTS_FOLDER.mkdir(exist_ok=True, parents=True)

IMAGE_FOLDER = ROOT_FOLDER / "images"
IMAGE_FOLDER.mkdir(exist_ok=True, parents=True)

DATA_FOLDER = ROOT_FOLDER / "data_store"
DATA_FOLDER.mkdir(exist_ok=True, parents=True)
