import os
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(Path(ROOT_DIR).parent.absolute(), 'dataset')

print(DATASET_PATH)