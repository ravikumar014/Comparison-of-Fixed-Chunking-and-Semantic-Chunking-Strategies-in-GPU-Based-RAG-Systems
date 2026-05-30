import json
from pathlib import Path


class ExperimentLogger:
    def __init__(self, filepath="results/ablation_results.json"):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        if self.filepath.exists():
            with open(self.filepath) as f:
                self.data = json.load(f)
        else:
            self.data = {}

    def log(self, experiment_name, result_dict):
        if experiment_name not in self.data:
            self.data[experiment_name] = []

        self.data[experiment_name].append(result_dict)

        with open(self.filepath, "w") as f:
            json.dump(self.data, f, indent=2)

    def reset_experiment(self, experiment_name):
        self.data[experiment_name] = []
        with open(self.filepath, "w") as f:
            json.dump(self.data, f, indent=2)