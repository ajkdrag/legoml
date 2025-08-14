from dataclasses import dataclass
from torch.utils.data import DataLoader
from legoml.core.base import DataLoaderNode


@dataclass
class DataLoaderForClassificationNode(DataLoaderNode):
    shuffle: bool = True

    def build(self, dataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
