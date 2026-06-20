"""
List extracted business concepts for a dataset.

Run from the DomainMiner root folder:

    python list_concepts.py -dataset_dir DellStore2

By default, the script reads:

    <dataset_folder>/ccm_output/step1_concepts.json

It prints the number of concepts and lists each concept on screen.
"""

import argparse
import json
import sys
from pathlib import Path


def load_concepts(path: Path) -> list:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        print(f"ERROR: concepts file not found: {path}")
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"ERROR: invalid JSON in {path}: {exc}")
        sys.exit(1)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("concepts", "business_concepts", "items"):
            value = data.get(key)
            if isinstance(value, list):
                return value

    print(f"ERROR: unsupported concepts format in {path}")
    sys.exit(1)


def concept_label(concept: object, index: int) -> tuple[str, str]:
    if not isinstance(concept, dict):
        return str(index), str(concept)

    concept_id = concept.get("id") or concept.get("concept_id") or str(index)
    name = concept.get("name") or concept.get("label") or concept.get("title")

    if not name:
        definition = concept.get("definition")
        name = definition if definition else "<unnamed concept>"

    return str(concept_id), str(name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count and list concepts extracted for a dataset."
    )
    parser.add_argument(
        "-dataset_dir",
        required=True,
        help="Dataset folder name, e.g. DellStore2, or a full path to the folder.",
    )
    parser.add_argument(
        "--concepts-file",
        default="ccm_output/step1_concepts.json",
        help="Concepts file relative to the dataset folder.",
    )
    args = parser.parse_args()

    candidate = Path(args.dataset_dir)
    dataset_dir = candidate if candidate.is_absolute() else Path.cwd() / candidate
    concepts_path = dataset_dir / args.concepts_file

    concepts = load_concepts(concepts_path)

    print(f"Dataset folder : {dataset_dir}")
    print(f"Concepts file  : {concepts_path}")
    print(f"Total concepts : {len(concepts)}")
    print()
    print("Extracted concepts:")

    for index, concept in enumerate(concepts, start=1):
        concept_id, name = concept_label(concept, index)
        print(f"  {index:>2}. {concept_id:<6} {name}")


if __name__ == "__main__":
    main()
