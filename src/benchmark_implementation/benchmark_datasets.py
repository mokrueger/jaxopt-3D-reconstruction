import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial

from src.config import DATASETS_PATH
from src.dataset.loaders.colmap_dataset_loader.loader import load_colmap_dataset


@dataclass
class DatasetConfig(Mapping):
    sparse_folder: str
    images_folder: str
    binary: bool
    name: str

    def __getitem__(self, item):
        return getattr(self, item)

    def __len__(self) -> int:
        return 4

    def __iter__(self):
        yield "sparse_folder"
        yield "images_folder"
        yield "binary"
        yield "name"


REICHSTAG_SPARSE_NOISED = os.path.join(DATASETS_PATH, "reichstag/sparse_noised")
REICHSTAG_SPARSE = os.path.join(DATASETS_PATH, "reichstag/sparse")
REICHSTAG_IMAGES = os.path.join(DATASETS_PATH, "reichstag/images")

SACRE_COEUR_SPARSE_NOISED = os.path.join(DATASETS_PATH, "sacre_coeur/sparse_noised")
SACRE_COEUR_IMAGES = os.path.join(DATASETS_PATH, "sacre_coeur/images")

ST_PETERS_SQUARE_SPARSE_NOISED = os.path.join(DATASETS_PATH, "st_peters_square/sparse_noised")
ST_PETERS_SQUARE_IMAGES = os.path.join(DATASETS_PATH, "st_peters_square/images")

""" Dataset Config Definitions """
REICHSTAG_NOISED_CONFIG = DatasetConfig(sparse_folder=REICHSTAG_SPARSE_NOISED,
                                        images_folder=REICHSTAG_IMAGES,
                                        binary=True,
                                        name="Reichstag")
REICHSTAG_GT_CONFIG = DatasetConfig(sparse_folder=REICHSTAG_SPARSE,
                                    images_folder=REICHSTAG_IMAGES,
                                    binary=True,
                                    name="Reichstag")
SACRE_COEUR_NOISED_CONFIG = DatasetConfig(sparse_folder=SACRE_COEUR_SPARSE_NOISED,
                                          images_folder=SACRE_COEUR_IMAGES,
                                          binary=True,
                                          name="Sacre Coeur")
ST_PETERS_NOISED_CONFIG = DatasetConfig(sparse_folder=ST_PETERS_SQUARE_SPARSE_NOISED,
                                        images_folder=ST_PETERS_SQUARE_IMAGES,
                                        binary=True,
                                        name="St Peters Square")


def partial_loader(sparse_folder, images_folder, binary, name):
    """ returns a partial function that loads the dataset """
    return partial(load_colmap_dataset, sparse_folder, images_folder, binary=binary, name=name)


""" PARTIAL LOADERS """
REICHSTAG_NOISED_LOADER = partial_loader(**REICHSTAG_NOISED_CONFIG)
REICHSTAG_GT_LOADER = partial_loader(**REICHSTAG_GT_CONFIG)
SACRE_COEUR_NOISED_LOADER = partial_loader(**SACRE_COEUR_NOISED_CONFIG)
ST_PETERS_SQUARE_NOISED_LOADER = partial_loader(**ST_PETERS_NOISED_CONFIG)
