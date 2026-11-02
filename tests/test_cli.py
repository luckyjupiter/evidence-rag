from __future__ import annotations

import json
from pathlib import Path

from evidence_rag.cli import main

CORPUS = Path(__file__).resolve().parents[1] / "examples" / "corpus"


def test_cli_ingest_search_and_stats(tmp_path, capsys):
    db = str(tmp_path / "cli.db")
    assert main(["--db", db, "ingest", str(CORPUS / "north-orchard-trial.md")]) == 0
    assert "ingested" in capsys.readouterr().out

    assert main(["--db", db, "search", "gravel layer", "--json"]) == 0
    hits = json.loads(capsys.readouterr().out)
    assert hits[0]["title"] == "North Orchard Drip Irrigation Trial"

    assert main(["--db", db, "stats", "--json"]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["documents"] == 1
    assert stats["chunks"] >= 1


def test_cli_reports_missing_file(tmp_path, capsys):
    db = str(tmp_path / "cli.db")
    assert main(["--db", db, "ingest", str(tmp_path / "nope.md")]) == 2
    assert "missing" in capsys.readouterr().err


def test_cli_rejects_title_for_multiple_paths(tmp_path):
    db = str(tmp_path / "cli.db")
    try:
        main(
            [
                "--db",
                db,
                "ingest",
                str(CORPUS / "north-orchard-trial.md"),
                str(CORPUS / "cedar-room-inventory.md"),
                "--title",
                "nope",
            ]
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:  # pragma: no cover
        raise AssertionError("expected argparse error")
