from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

PROMPT = """\
Parse the raw data file and extract the relevant information for every GPU mentioned in the file. \
Put null if the information is not mentioned in the source.

### BEGIN SOURCE ###
{raw_data}
### END SOURCE ###

Return a YAML file with the following structure (use the exact same keys but not the values, they are just an example):
```yaml
- gpu: T4
  architecture: Turing
  code_name:
    - TU104-895-A1
  l2_cache:
    value: 4
    unit: MB
  vram:
    value: 16
    unit: GB
  memory_bandwidth:
    value: 320
    unit: GB/s
  cuda_cores:
    value: 2560
    unit: null
  sm_count:
    value: 48
    unit: null
  max_clock:
    value: 1590
    unit: MHz
- ...
```
"""


def parse_raw_data(raw_data_file: Path) -> str:
    with open(raw_data_file) as f:
        raw_data = f.read()

    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": PROMPT.format(raw_data=raw_data),
            },
        ],
        temperature=0.5,
        top_p=0.95,
    )
    out = completion.choices[0].message.content
    assert (
        isinstance(out, str) and out.startswith("```yaml\n") and out.endswith("\n```")
    )
    out = out.removeprefix("```yaml\n").removesuffix("\n```")
    return out


if __name__ == "__main__":
    load_dotenv()

    gpu_dir = Path("data", "gpus")
    raw_data_sources = (gpu_dir / "raw").glob("*.*")
    out_dir = gpu_dir / "parsed"
    out_dir.mkdir(exist_ok=True, parents=True)

    for raw_data_file in raw_data_sources:
        print(f"Parsing {raw_data_file}")
        parsed_data = parse_raw_data(raw_data_file)

        with open(out_dir / f"{raw_data_file.stem}.yaml", "w") as f:
            f.write(parsed_data)
