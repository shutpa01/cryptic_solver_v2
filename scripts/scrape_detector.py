"""Scrape detector — flag IPs that look like daily clue-page scrapers.

Reads nginx access logs from the production droplet over a recent
window and reports IPs whose request pattern matches batch scraping:
many /clue/<slug> hits in a short window, single User-Agent.

Important caveats given the current Cloudflare + nginx setup:

  - The IPs shown are Cloudflare proxy IPs, not the real client IPs.
    nginx isn't currently configured to log X-Forwarded-For, so we
    only see the CF edge node that routed each request. A single
    real client may appear across several CF IPs, which under-
    counts. To fix properly, add `$http_x_forwarded_for` to the
    nginx log_format or enable the ngx_http_realip_module.
  - Static asset hits are always zero in this view because CF
    serves CSS/JS/images from its cache and they never reach
    origin. So the "asset/clue" ratio heuristic that would normally
    flag scrapers (HTML-only fetch) is unusable here.

Despite both, the report still highlights useful patterns: high
request rate, narrow time concentration, and single-UA bursts are
all visible per CF IP and worth investigating.

Usage:
    python scripts/scrape_detector.py
    python scripts/scrape_detector.py --days 7 --threshold 30
"""

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

DROPLET = "root@165.232.46.255"
LOG_FILES_PLAIN = ["/var/log/nginx/access.log", "/var/log/nginx/access.log.1"]

# Subprocess ssh on Windows picks a different ssh binary than the one Bash
# uses, with Unix-style default paths that don't resolve under Windows HOME.
# Force it at the user's real known_hosts and identity file.
import os as _os
_SSH_DIR = _os.path.join(_os.path.expanduser("~"), ".ssh")
_KNOWN_HOSTS = _os.path.join(_SSH_DIR, "known_hosts")
_IDENTITY = _os.path.join(_SSH_DIR, "id_rsa")

# Static asset paths — a real browser fetches these alongside the HTML;
# a scraper usually doesn't.
ASSET_RE = re.compile(
    r"\.(?:css|js|png|jpe?g|gif|svg|ico|woff2?|ttf|webp|map)(?:\?|$)",
    re.IGNORECASE,
)
CLUE_RE = re.compile(r"^/clue/")
LOG_LINE_RE = re.compile(
    r'^(\S+) \S+ \S+ \[([^\]]+)\] "(\S+) (\S+) [^"]+" (\d+) (\d+|-) '
    r'"([^"]*)" "([^"]*)"'
)
NGINX_TIME = "%d/%b/%Y:%H:%M:%S %z"


def fetch_log_lines(days: int) -> list[str]:
    """Pull lines from droplet logs covering today and the last `days` days."""
    today = datetime.now(timezone.utc)
    dates = [(today - timedelta(days=d)).strftime("%d/%b/%Y") for d in range(days + 1)]
    pattern = "|".join(re.escape(d) for d in dates)
    grep_cmd = f"grep -h -E '{pattern}' " + " ".join(LOG_FILES_PLAIN) + " 2>/dev/null"
    # -o StrictHostKeyChecking=accept-new: avoid an interactive prompt the
    # first time this runs.
    # -o UserKnownHostsFile / IdentityFile: subprocess on Windows otherwise
    # looks at Unix-style /home/<user>/.ssh paths and fails to find the
    # existing key material.
    ssh_opts = [
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", f"UserKnownHostsFile={_KNOWN_HOSTS}",
        "-o", f"IdentityFile={_IDENTITY}",
        "-o", "IdentitiesOnly=yes",
    ]
    res = subprocess.run(
        ["ssh", *ssh_opts, DROPLET, grep_cmd],
        capture_output=True, text=True, timeout=180,
    )
    return res.stdout.splitlines()


def parse_line(line: str) -> dict | None:
    m = LOG_LINE_RE.match(line)
    if not m:
        return None
    try:
        ts = datetime.strptime(m.group(2), NGINX_TIME)
    except ValueError:
        return None
    return {
        "ip": m.group(1),
        "ts": ts,
        "path": m.group(4),
        "status": int(m.group(5)),
        "ua": m.group(8),
    }


def analyse(records: list[dict], threshold: int) -> list[dict]:
    by_ip: dict[str, dict] = defaultdict(
        lambda: {"clue_hits": 0, "asset_hits": 0, "other_hits": 0,
                 "uas": set(), "times": [], "clue_paths": []}
    )
    for r in records:
        d = by_ip[r["ip"]]
        d["uas"].add(r["ua"])
        if CLUE_RE.match(r["path"]):
            d["clue_hits"] += 1
            d["times"].append(r["ts"])
            d["clue_paths"].append(r["path"])
        elif ASSET_RE.search(r["path"]):
            d["asset_hits"] += 1
        else:
            d["other_hits"] += 1

    candidates = []
    for ip, d in by_ip.items():
        if d["clue_hits"] < threshold:
            continue
        # Time span and burstiness
        times = sorted(d["times"])
        span_s = (times[-1] - times[0]).total_seconds() if len(times) > 1 else 0
        rate_per_min = d["clue_hits"] / max(span_s / 60, 1)
        # Asset/HTML ratio — a real browser is usually 5x+ assets per HTML
        ratio = d["asset_hits"] / max(d["clue_hits"], 1)
        # Hour-of-day distribution (UTC)
        hour_counts: dict[int, int] = defaultdict(int)
        for t in times:
            hour_counts[t.hour] += 1
        peak_hour = max(hour_counts, key=hour_counts.get) if hour_counts else None
        peak_share = hour_counts[peak_hour] / d["clue_hits"] if peak_hour is not None else 0

        # Suspicion score — tunable, ranks the report.
        # asset_ratio is unusable in the CF-proxied setup (assets cached
        # at the edge never hit origin), so we don't score on it.
        score = 0
        if peak_share > 0.8:
            score += 3      # Concentrated in one hour — strong batch signal
        if span_s > 0 and rate_per_min > 10:
            score += 2      # Fast, sustained — looks automated
        elif span_s > 0 and rate_per_min > 5:
            score += 1
        if len(d["uas"]) == 1:
            score += 1
        if d["clue_hits"] > 100 and span_s < 600:
            score += 2      # 100+ hits in <10 min is a tight batch

        candidates.append({
            "ip": ip,
            "clue_hits": d["clue_hits"],
            "asset_hits": d["asset_hits"],
            "other_hits": d["other_hits"],
            "asset_ratio": ratio,
            "span_seconds": span_s,
            "rate_per_min": rate_per_min,
            "peak_hour": peak_hour,
            "peak_share": peak_share,
            "first_time": times[0] if times else None,
            "last_time": times[-1] if times else None,
            "uas": list(d["uas"]),
            "score": score,
        })
    return sorted(candidates, key=lambda c: (c["score"], c["clue_hits"]), reverse=True)


def report(candidates: list[dict]) -> None:
    if not candidates:
        print("No IPs matched the threshold. No scrape pattern detected.")
        return
    print(f"{'CF-IP':<18} {'clue':>5} {'rt/m':>5} {'span':>8} "
          f"{'peakH':>5} {'peak%':>5} {'UAs':>3} score")
    print("-" * 64)
    for c in candidates:
        span_str = f"{c['span_seconds']/60:.1f}m" if c['span_seconds'] < 3600 else f"{c['span_seconds']/3600:.1f}h"
        peak_str = str(c['peak_hour']) if c['peak_hour'] is not None else '-'
        print(f"{c['ip']:<18} {c['clue_hits']:>5} "
              f"{c['rate_per_min']:>5.1f} {span_str:>8} "
              f"{peak_str:>5} {c['peak_share']*100:>4.0f}% "
              f"{len(c['uas']):>3} {c['score']:>5}")
    print()
    # Detail block for each top scoring candidate
    top = [c for c in candidates if c["score"] >= 5]
    if not top:
        print("(No high-score candidates for detail. Lower threshold or wait for more data.)")
        return
    print(f"=== Detail for top-scoring candidates (score >= 5) ===")
    print()
    for c in top:
        print(f"-- {c['ip']}  (score {c['score']}) --")
        print(f"   clue hits      : {c['clue_hits']}")
        print(f"   first / last   : {c['first_time']}  ->  {c['last_time']}")
        print(f"   peak hour (UTC): {c['peak_hour']}  ({c['peak_share']*100:.0f}% of hits)")
        print(f"   rate per minute: {c['rate_per_min']:.1f}")
        print(f"   user-agents ({len(c['uas'])}):")
        for ua in c["uas"][:5]:
            print(f"     - {ua[:120]}")
        print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=1,
                    help="Look back this many days (default: 1)")
    ap.add_argument("--threshold", type=int, default=30,
                    help="Minimum /clue/ hits per IP to consider (default: 30)")
    args = ap.parse_args()

    print(f"Fetching last {args.days} day(s) of /clue/-relevant logs from droplet...")
    lines = fetch_log_lines(args.days)
    print(f"Pulled {len(lines):,} log lines.")
    if not lines:
        print("No log lines retrieved. Check SSH access and log paths.")
        return 1

    records = [r for r in (parse_line(l) for l in lines) if r is not None]
    print(f"Parsed {len(records):,} records.")

    candidates = analyse(records, args.threshold)
    report(candidates)
    return 0


if __name__ == "__main__":
    sys.exit(main())
