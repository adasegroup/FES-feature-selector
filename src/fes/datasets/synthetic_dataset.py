from typing import Dict, Any
from kedro.io.core import AbstractDataSet


class SyntheticDataset(AbstractDataSet):
    def __init__(self):
        pass

    def _load(self) -> Any:
        pass

    def _save(self, data: Any) -> None:
        pass

    def _describe(self) -> Dict[str, Any]:
        pass
