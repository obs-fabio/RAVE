import os
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


METRIC = "multiband_spectral_distance"


def find_event_files(base_dir):
    return sorted(Path(base_dir).rglob("events.out.tfevents.*"))


def load_metric(event_file):

    ea = EventAccumulator(
        str(event_file),
        size_guidance={"scalars": 0},
    )
    ea.Reload()

    if METRIC not in ea.Tags()["scalars"]:
        return None

    events = ea.Scalars(METRIC)

    return pd.DataFrame(
        {
            "step": [e.step for e in events],
            "value": [e.value for e in events],
        }
    )

def tensorboard_smoothing(values, smoothing=0.9):
    values = list(values)

    if len(values) == 0:
        return values

    smoothed = [values[0]]

    for value in values[1:]:
        smoothed.append(
            smoothing * smoothed[-1] +
            (1 - smoothing) * value
        )

    return smoothed

def main(base_dir):

    base_dir = Path(base_dir)
    output_dir = "./runs/"
    batch_size = 55
    smoothing = 0.9

    df_all = None

    for i, event_file in enumerate(find_event_files(base_dir)):

        model = event_file.relative_to(base_dir).parts[0]

        print(f"{model}: {event_file}")

        try:
            df = load_metric(event_file)
        except Exception:
            continue

        if df is None:
            continue

        # renomeia a coluna "value" para o nome do modelo
        df = df.rename(columns={"value": model})

        if df_all is None:
            df_all = df
        else:
            df_all = df_all.merge(df, on="step", how="outer")

    if df_all is None:
        print("Nenhum dado encontrado.")
        return

    df_all = df_all.sort_values("step")

    csv_name = f"{METRIC}.csv"
    df_all.to_csv(os.path.join(output_dir, csv_name), index=False)

    print(df_all.head())


    plt.figure(figsize=(12,6))

    for col in df_all.columns:

        if col == "step":
            continue

        y = tensorboard_smoothing(df_all[col].values, smoothing)

        plt.plot(
            df_all["step"]/batch_size,
            y,
            label=col,
            linewidth=2,
        )

    plt.xlabel("Batch")
    plt.ylabel(METRIC)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{METRIC}.png"), dpi=300)
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir")
    args = parser.parse_args()

    main(args.base_dir)
