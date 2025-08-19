from abc import ABC, abstractmethod
from legoml.core.step_output import StepOutput


class Metric(ABC):
    name: str

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self, output: StepOutput):
        pass

    @abstractmethod
    def compute(self) -> dict[str, float]:
        pass
