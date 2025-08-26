import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass


from propagator.models import PropagatorStats
from propagator_io.writer.protocol import MetadataWriterProtocol

@dataclass
class MetadataJSONWriter(MetadataWriterProtocol):
    output_folder: Path
    prefix: str

    def write_metadata(
            self,
            stats: PropagatorStats, 
            c_time: int,
            ref_date: datetime
        ) -> None:
        json_file = self.output_folder / f"{self.prefix}_{c_time}.json"
        with open(json_file, "w") as fp:
            data = stats.to_dict(c_time, ref_date)
            json.dump(data, fp)
