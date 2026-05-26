import queue
import sqlite3
import sys
import threading
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLUES_DB = PROJECT_ROOT / "data" / "clues_master.db"
TIMEOUT_SECONDS = 30
SOURCE = "dailymail"
PUZZLE_NUMBER = "17884"

sys.path.insert(0, str(PROJECT_ROOT))

from signature_solver.db import RefDB
from sonnet_pipeline.clue_pipeline import run_signature_clue_pipeline


def load_tests(conn):
    rows = conn.execute(
        """
        SELECT id, clue_number, direction, clue_text, answer, source, puzzle_number
        FROM clues
        WHERE source = ?
          AND puzzle_number = ?
        ORDER BY CASE direction WHEN 'across' THEN 0 ELSE 1 END,
                 CAST(clue_number AS INTEGER),
                 clue_number,
                 id
        """,
        (SOURCE, PUZZLE_NUMBER),
    ).fetchall()

    return [
        {
            "clue_id": clue_id,
            "clue_number": clue_number,
            "direction": direction,
            "clue_text": clue_text,
            "answer": answer,
            "source": source,
            "puzzle_number": puzzle_number,
        }
        for (
            clue_id,
            clue_number,
            direction,
            clue_text,
            answer,
            source,
            puzzle_number,
        ) in rows
    ]


def run_pipeline_worker(test, ref_db, result_queue):
    conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    try:
        result = run_signature_clue_pipeline(
            conn,
            test["clue_id"],
            test["source"],
            test["puzzle_number"],
            test["clue_text"],
            test["answer"],
            ref_db,
            write_db=False,
        )
        result_queue.put(("ok", result))
    except Exception as exc:
        result_queue.put(("error", exc))
    finally:
        conn.close()


def confidence_value(solve_result):
    if solve_result is None:
        return None
    return getattr(solve_result, "confidence", None)


def is_high_confidence(solve_result):
    confidence = confidence_value(solve_result)
    return confidence is not None and confidence >= 80


def stage_two_path(solve_result, high_conf):
    if solve_result is None:
        return "none"

    stage_two = getattr(solve_result, "stage_two_casefile", None)
    if stage_two is None:
        return "none"

    if getattr(stage_two, "status", None) == "answer_fit" and high_conf is True:
        return "from_solve_result"

    if high_conf is False:
        return "grammar_only"

    return "unknown"


def stage_three_status(solve_result):
    if solve_result is None:
        return "none"

    stage_three = getattr(solve_result, "stage_three_proof", None)
    if stage_three is None:
        return "none"

    return getattr(stage_three, "status", "none") or "none"


def mechanism_label(test):
    direction = (test["direction"] or "")[:1].lower()
    return f"{test['clue_number']}{direction}"


def run_one_test(test, ref_db):
    result_queue = queue.Queue(maxsize=1)
    start = time.time()
    thread = threading.Thread(
        target=run_pipeline_worker,
        args=(test, ref_db, result_queue),
        daemon=True,
    )
    thread.start()

    try:
        status, payload = result_queue.get(timeout=TIMEOUT_SECONDS)
        runtime = time.time() - start
        timed_out = False
    except queue.Empty:
        runtime = time.time() - start
        timed_out = True
        status = "timeout"
        payload = None

    solve_result = None
    error = None
    if status == "ok":
        solve_result = getattr(payload, "solve_result", None)
    elif status == "error":
        error = payload

    high_conf = is_high_confidence(solve_result)
    return {
        "answer": test["answer"],
        "mechanism": mechanism_label(test),
        "confidence": confidence_value(solve_result),
        "s2_path": stage_two_path(solve_result, high_conf),
        "s3_status": stage_three_status(solve_result),
        "runtime": runtime,
        "timed_out": timed_out,
        "result": "FAIL" if timed_out else "PASS",
        "error": error,
    }


def format_confidence(row):
    if row["confidence"] is None:
        return "none"
    return str(row["confidence"])


def print_table(rows):
    headers = ["ANSWER", "MECHANISM", "CONF", "S2_PATH", "S3_STATUS", "TIME", "RESULT"]
    data = []
    for row in rows:
        result = row["result"]
        if row["error"] is not None:
            result = f"{result}: {type(row['error']).__name__}"
        data.append(
            [
                row["answer"],
                row["mechanism"],
                format_confidence(row),
                row["s2_path"],
                row["s3_status"],
                f"{row['runtime']:.2f}s",
                result,
            ]
        )

    widths = [
        max(len(str(item)) for item in [header] + [row[index] for row in data])
        for index, header in enumerate(headers)
    ]

    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in data:
        print("  ".join(str(item).ljust(widths[index]) for index, item in enumerate(row)))


def print_summary(rows):
    total = len(rows)
    timed_out = sum(1 for row in rows if row["timed_out"])
    passed = total - timed_out
    print()
    print(f"{passed}/{total} passed ({timed_out} timed out)")


def main():
    conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    try:
        tests = load_tests(conn)
    finally:
        conn.close()

    ref_db = RefDB()
    rows = [run_one_test(test, ref_db) for test in tests]
    print_table(rows)
    print_summary(rows)


if __name__ == "__main__":
    main()
