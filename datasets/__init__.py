from .aircraft_dataset import AircraftDataset
from .bird_dataset import BirdDataset
from .car_dataset import CarDataset
from .fish_dataset import FishDataset


def get_trainval_datasets(tag, resize, fold: int):
    if tag == 'aircraft':
        return AircraftDataset(phase='train', resize=resize), AircraftDataset(phase='val', resize=resize)
    elif tag == 'bird':
        return BirdDataset(phase='train', resize=resize), BirdDataset(phase='val', resize=resize)
    elif tag == 'car':
        return CarDataset(phase='train', resize=resize), CarDataset(phase='val', resize=resize)
    elif tag == 'fish':
        return FishDataset(phase='train', resize=resize, fold=fold), FishDataset(phase='val', resize=resize, fold=fold)
    else:
        raise ValueError('Unsupported Tag {}'.format(tag))


def get_fish_test_dataset(resize):
    return FishDataset(phase='test', resize=resize, fold=-1)
