import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy.typing as npt
import numpy as np

from propagator_io.writer.protocol import IsochronesWriterProtocol


@dataclass
class IsochronesGeoJSONWriter(IsochronesWriterProtocol):
    output_folder: Path
    prefix: str

    def write_isochrones(
        self,
        isochrones: dict[int, npt.NDArray[np.floating]],
        c_time: int,
        ref_date: datetime
    ) -> None:
        json_file = self.output_folder / f"{self.prefix}_{c_time}.json"
        with open(json_file, "w") as fp:
            json.dump(isochrones, fp)
