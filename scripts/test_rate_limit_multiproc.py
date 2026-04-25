"""Multi-process rate limiter test.

Spawns N parallel worker processes, each hitting the same simulated IP
via web/rate_limit._check_and_increment. Sums the per-worker allowed
counts; result should match the configured limit, NOT N x limit.

Usage:
    python scripts/test_rate_limit_multiproc.py
"""

import json
import multiprocessing as mp
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def worker(args):
    worker_id, scope, ip, limit, window, requests_each = args
    from web.rate_limit import _check_and_increment

    allowed = 0
    blocked = 0
    for _ in range(requests_each):
        ok, _retry = _check_and_increment(scope, ip, limit, window)
        if ok:
            allowed += 1
        else:
            blocked += 1
    return {"worker_id": worker_id, "allowed": allowed, "blocked": blocked}


def main():
    from web.rate_limit import _testing_reset

    scope = "multiproc_test"
    ip = "198.51.100.123"
    limit = 30
    window = 60
    n_workers = 4
    requests_each = 25  # total = 100, well above limit

    _testing_reset()

    args = [
        (i, scope, ip, limit, window, requests_each) for i in range(n_workers)
    ]

    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(worker, args)

    total_allowed = sum(r["allowed"] for r in results)
    total_blocked = sum(r["blocked"] for r in results)
    total_requests = total_allowed + total_blocked

    print(f"Configured limit: {limit}/{window}s")
    print(f"Workers: {n_workers}, requests per worker: {requests_each}")
    print(f"Total requests: {total_requests}")
    print(f"Total ALLOWED: {total_allowed}  (should equal limit = {limit})")
    print(f"Total BLOCKED: {total_blocked}")
    print()
    print("Per-worker breakdown:")
    for r in sorted(results, key=lambda x: x["worker_id"]):
        print(f"  worker {r['worker_id']}: allowed={r['allowed']}, blocked={r['blocked']}")
    print()

    if total_allowed == limit:
        print("PASS: cross-process limit holds at exactly the configured value.")
        sys.exit(0)
    else:
        print(f"FAIL: expected {limit} total allowed, got {total_allowed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
