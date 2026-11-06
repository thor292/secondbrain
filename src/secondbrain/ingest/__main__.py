"""Ingestion admin commands (admin connection).

    uv run python -m secondbrain.ingest schema                      # add the ingest tables to an existing database
    uv run python -m secondbrain.ingest key create desiliving "Desi Living ERP" [--scope merchandising]
    uv run python -m secondbrain.ingest key list [desiliving]
    uv run python -m secondbrain.ingest key revoke <key id>
    uv run python -m secondbrain.ingest csv desiliving merchandising daily_sales history.csv
    uv run python -m secondbrain.ingest errors [desiliving]
    uv run python -m secondbrain.ingest refresh desiliving            # recalculate forecasts and signals now
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import psycopg

from secondbrain.config import get_settings
from secondbrain.ingest import backfill, contract, keys, processor, store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("schema")
    key = sub.add_parser("key").add_subparsers(dest="action", required=True)
    create = key.add_parser("create")
    create.add_argument("company")
    create.add_argument("name")
    create.add_argument("--scope", help="department id or path; default: the whole company")
    key.add_parser("list").add_argument("company", nargs="?")
    key.add_parser("revoke").add_argument("key_id")
    csv_cmd = sub.add_parser("csv")
    for name in ("company", "brain", "kind", "file"):
        csv_cmd.add_argument(name)
    sub.add_parser("errors").add_argument("company", nargs="?")
    sub.add_parser("refresh").add_argument("company")
    args = parser.parse_args()

    settings = get_settings()
    with psycopg.connect(settings.admin_database_url, autocommit=True) as conn:
        store.ensure_schema(conn)
        if args.command == "schema":
            print("Ingest tables ready.")
        elif args.command == "key" and args.action == "create":
            key_id, secret = keys.create(conn, args.company, args.name, args.scope)
            print(f"Key {key_id} created. Give this to the client once; it can't be shown again:\n{secret}")
        elif args.command == "key" and args.action == "list":
            for k in keys.listing(conn, args.company):
                state = f"revoked {k['revoked_at']:%Y-%m-%d}" if k["revoked_at"] else "active"
                print(f"{k['id']}  {k['prefix']}…  {k['scope']:<30} {k['name']} ({state})")
        elif args.command == "key":
            print("Revoked." if keys.revoke(conn, args.key_id) else "No such active key.")
        elif args.command == "csv":
            company = contract.company_path(args.company)
            items = backfill.messages(args.kind, Path(args.file).read_text(encoding="utf-8"), company, args.brain)
            outcomes = [processor.process(conn, args.kind, item, "csv") for item in items]
            print(dict(Counter(o.status for o in outcomes)))
            for o in [o for o in outcomes if o.status == "rejected"][:20]:
                print(f"  {o.event_id}: {o.error}")
        elif args.command == "errors":
            for r in store.dead_letters(conn, args.company):
                print(f"{r[0]:%Y-%m-%d %H:%M}  {r[1]}  {r[3]}  {r[2]}\n  {r[7]}\n  {json.dumps(r[8])[:300]}")
    if args.command == "refresh":
        from secondbrain.ingest.pipeline import refresh_company
        from secondbrain.models import build_models

        refresh_company(settings, *build_models(settings), contract.company_path(args.company))
        print("Forecasts and signals refreshed.")


if __name__ == "__main__":
    main()
