import re

import numpy as np
import pandas as pd
import yaml

SOURCES = [
    {
        "path": "data/gpus/parsed/2000_series_desktop.yaml",
        "overrides": [
            {
                "pattern": r".*",
                "architecture": "Turing",
                "tensor_core_type": "consumer",
            },
            {
                "pattern": r".*Titan.*",
                "architecture": "Turing",
                "tensor_core_type": "datacenter",
            },
        ],
    },
    {
        "path": "data/gpus/parsed/2000_series_mobile.yaml",
        "overrides": [
            {
                "pattern": r".*",
                "architecture": "Turing",
                "tensor_core_type": "consumer",
            }
        ],
    },
    {
        "path": "data/gpus/parsed/3000_series_desktop.yaml",
        "overrides": [
            {
                "pattern": r".*",
                "architecture": "Ampere",
                "tensor_core_type": "consumer",
            }
        ],
    },
    {
        "path": "data/gpus/parsed/3000_series_mobile.yaml",
        "overrides": [
            {
                "pattern": r".*",
                "architecture": "Ampere",
                "tensor_core_type": "consumer",
            }
        ],
    },
    {
        "path": "data/gpus/parsed/4000_series_desktop.yaml",
        "overrides": [
            {
                "pattern": r".*",
                "architecture": "Ada Lovelace",
                "tensor_core_type": "consumer",
            }
        ],
    },
    {
        "path": "data/gpus/parsed/4000_series_mobile.yaml",
        "overrides": [
            {
                "pattern": r".*",
                "architecture": "Ada Lovelace",
                "tensor_core_type": "consumer",
            }
        ],
    },
    {
        "path": "data/gpus/manual/a100.yaml",
        "overrides": [
            {
                "pattern": r".*",
                "tensor_core_type": "datacenter",
            }
        ],
    },
]


def gpu_table():
    data = []
    for src in SOURCES:
        with open(src["path"]) as f:
            gpu_data = yaml.safe_load(f)

        for gpu in gpu_data:
            for override in src["overrides"]:
                override_ = override.copy()
                pattern = override_.pop("pattern")
                if re.fullmatch(pattern, gpu["gpu"]) is not None:
                    for key, value in override_.items():
                        gpu[key] = value
            for key, value in gpu.items():
                match value:
                    case {"value": a, "unit": b}:
                        gpu[key] = a
                    case _:
                        pass
        data += gpu_data
    df = pd.DataFrame(data)
    return df


def tensor_core_table():
    with open("data/tensor_cores.yaml") as f:
        tensor_cores_data = yaml.safe_load(f)

    tc_data = []
    for tc in tensor_cores_data:
        for fma in tc["fma_per_sm_per_cycle"]:
            tc_data.append(
                {
                    "tensor_core_gen": tc["tensor_core_gen"],
                    "tensor_core_type": tc["tensor_core_type"],
                    "input": fma["input"],
                    "accumulation": fma["accumulation"],
                    "value": fma["value"],
                }
            )
    df = pd.DataFrame(tc_data)
    df["input_accumulation"] = df["input"] + "x" + df["accumulation"]
    df = df.pivot(
        index=["tensor_core_gen", "tensor_core_type"],
        columns="input_accumulation",
        values="value",
    )
    return df


def tensor_core_mapping_table():
    with open("data/tensor_cores_arch_to_gen.yaml") as f:
        arch_to_gen = yaml.safe_load(f)

    df = pd.DataFrame(arch_to_gen.items(), columns=["architecture", "tensor_core_gen"])
    df.set_index("architecture", inplace=True)
    return df


if __name__ == "__main__":
    gpu_df = gpu_table()
    tc_df = tensor_core_table()
    tc_mapping = tensor_core_mapping_table()

    # Join the tables
    df = gpu_df.merge(tc_mapping, on="architecture")
    df = df.merge(tc_df, on=["tensor_core_gen", "tensor_core_type"])

    # Calculate the tensor core performance
    for col in tc_df.columns:
        df[col] *= 2 * df["sm_count"] * df["max_clock"] / 1e6

    # print(
    #     df[
    #         ["gpu", "architecture", "tensor_core_type", "max_clock", "sm_count", *tc_df.columns]
    #     ]  # .query("gpu.str.contains('3080 Ti')")
    #     .query("gpu.str.contains('Titan')")
    #     .round(1)
    # )

    # Pretty print the table
    df[tc_df.columns] = df[tc_df.columns].round(1)
    df[df[tc_df.columns] < 0] = "?"
    df = df.replace(np.NaN, "-")
    arch_order = [
        "Volta",
        "Turing",
        "Ampere",
        "Ada Lovelace",
        "Hopper",
    ]
    df["architecture"] = pd.Categorical(
        df["architecture"], categories=arch_order, ordered=True
    )
    df = df.sort_values(["architecture", "sm_count"])
    print(df.to_markdown(index=False))
