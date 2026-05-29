import json
import tempfile
from pathlib import Path

def test_split_ratio():
    from data.scripts.split_dataset import split_dataset
    samples = [{"id": i} for i in range(100)]
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "data.json"
        with open(input_path, "w") as f:
            json.dump(samples, f)
        split_dataset(str(input_path), tmpdir, train_ratio=0.8, val_ratio=0.1)
        train = json.loads((Path(tmpdir) / "train.json").read_text())
        val = json.loads((Path(tmpdir) / "val.json").read_text())
        test = json.loads((Path(tmpdir) / "test.json").read_text())
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10
