"""Regenerate all human-readable outputs from the frozen JSON artifact."""
from artifact_pipeline import generate, load_artifact


def main():
    artifact = load_artifact()
    generate(artifact)
    print("regenerated RESULTS_SUMMARY.md, CSV, figures, and LaTeX fragments")


if __name__ == "__main__":
    main()
