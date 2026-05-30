import json
from pathlib import Path

def save_metrics(name, metrics):
    Path("results").mkdir(exist_ok=True)

    with open("results/metrics.json", "a") as f:
        json.dump({name: metrics}, f)
        f.write("\n")