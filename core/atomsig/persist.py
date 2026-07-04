"""Persist harvested atom-map signatures (logs/atomsig/*.jsonl) into data/atomsig.db.

Reads the SOUND harvester output and writes a real catalogue:
  signature(key, count, example_clue, example_answer)
  instance(clue_id, signature, assembly, def_pos, operation, info_json)

Idempotent: drops + recreates the two tables on each run. Does NOT touch any
other DB. Run:  .venv/Scripts/python.exe -m core.atomsig.persist
"""
import json
import os
import sqlite3

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SIG_PATH = os.path.join(ROOT, "logs", "atomsig", "signatures.jsonl")
INST_PATH = os.path.join(ROOT, "logs", "atomsig", "instances.jsonl")
DB_PATH = os.path.join(ROOT, "data", "atomsig.db")


def _read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    sigs = _read_jsonl(SIG_PATH)
    insts = _read_jsonl(INST_PATH)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.executescript(
        """
        DROP TABLE IF EXISTS signature;
        DROP TABLE IF EXISTS instance;
        CREATE TABLE signature (
            key TEXT PRIMARY KEY,
            count INTEGER,
            example_clue TEXT,
            example_answer TEXT
        );
        CREATE TABLE instance (
            clue_id INTEGER,
            signature TEXT,
            assembly TEXT,
            def_pos TEXT,
            operation TEXT,
            info_json TEXT
        );
        """
    )

    for s in sigs:
        ex = s.get("example") or {}
        cur.execute(
            "INSERT OR REPLACE INTO signature(key, count, example_clue, example_answer) VALUES (?,?,?,?)",
            (s["signature"], s.get("count"), ex.get("clue"), ex.get("answer")),
        )

    for r in insts:
        assembly = r["signature"].split(" | ", 1)[0].strip()
        cur.execute(
            "INSERT INTO instance(clue_id, signature, assembly, def_pos, operation, info_json) VALUES (?,?,?,?,?,?)",
            (
                r.get("id"),
                r.get("signature"),
                assembly,
                r.get("def_pos"),
                r.get("operation"),
                json.dumps(r),
            ),
        )

    cur.execute("CREATE INDEX idx_instance_signature ON instance(signature)")
    conn.commit()

    sig_n = cur.execute("SELECT COUNT(*) FROM signature").fetchone()[0]
    inst_n = cur.execute("SELECT COUNT(*) FROM instance").fetchone()[0]
    print(f"data/atomsig.db written")
    print(f"  signature rows: {sig_n}")
    print(f"  instance  rows: {inst_n}")
    print()
    print("Top 20 signatures by count:")
    print(f"  {'count':>5}  signature")
    for key, cnt in cur.execute(
        "SELECT key, count FROM signature ORDER BY count DESC, key LIMIT 20"
    ):
        print(f"  {cnt:>5}  {key}")

    conn.close()


if __name__ == "__main__":
    main()
