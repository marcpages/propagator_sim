import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass


from propagator.models import PropagatorOutput
from propagator_io.writer.protocol import MetadataWriterProtocol


@dataclass
class MetadataJSONWriter(MetadataWriterProtocol):
    start_date: datetime
    output_folder: Path
    prefix: str

    def write_metadata(self, output: PropagatorOutput) -> None:
        ref_date = self.ref_date(output)
        json_file = self.output_folder / f"{self.prefix}_{output.time}.json"
        with open(json_file, "w") as fp:
            data = output.stats.to_dict(output.time, ref_date)
            json.dump(data, fp)
