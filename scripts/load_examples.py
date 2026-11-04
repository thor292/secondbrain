"""Load the example companies: onboard them and import their documents.

Each company folder has a company.json and docs/<brain>/<sensitivity>/ with
the documents for that brain ("company" means the company brain itself).
To demo with a real company, give it the same layout.

    uv run python scripts/load_examples.py
"""

import json
from pathlib import Path

import psycopg

from secondbrain import admin
from secondbrain.config import get_settings
from secondbrain.models import build_models
from secondbrain.sensitivity import Sensitivity

EXAMPLES = Path(__file__).parent.parent / "examples"


def main() -> None:
    settings = get_settings()
    _, embedder = build_models(settings)
    with psycopg.connect(settings.admin_database_url) as conn:
        for spec_path in sorted(EXAMPLES.glob("*/company.json")):
            company = admin.onboard(conn, json.loads(spec_path.read_text()))
            print(f"Onboarded {company}")
            for folder in sorted((spec_path.parent / "docs").glob("*/*")):
                brain = company if folder.parent.name == "company" else f"{company}.{folder.parent.name}"
                documents = admin.read_documents(folder)
                if documents:
                    count = admin.import_documents(
                        conn, embedder, settings, brain, documents, Sensitivity.parse(folder.name)
                    )
                    print(f"  {brain:<28} {folder.name:<10} {len(documents)} documents, {count} chunks")
            conn.commit()


if __name__ == "__main__":
    main()
