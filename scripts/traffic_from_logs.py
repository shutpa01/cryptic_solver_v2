"""Count real visitors from the web server's own access logs.

Ground truth for "how many people were on the site today", independent of
GA4 (which lags up to 24h), Bing Webmaster Tools (which stalls), and
Clarity (whose headline card is page views, not people).

Reads nginx / gunicorn combined-format logs, drops bots and static
assets, and reports page views and distinct visitors per day, optionally
split by where the traffic came from.

Usage:
    python3 scripts/traffic_from_logs.py                       # auto-detect, 7 days
    python3 scripts/traffic_from_logs.py --days 1              # today only
    python3 scripts/traffic_from_logs.py --by-source           # referrer split
    python3 scripts/traffic_from_logs.py --top-pages 10        # busiest pages
    python3 scripts/traffic_from_logs.py /var/log/nginx/access.log*
    python3 scripts/traffic_from_logs.py --bots                # what the crawlers did

Caveats it will not hide from you:
  - A visitor is a distinct (IP, user-agent) pair. Mobile networks share
    IPs and people switch devices, so this is an estimate, like every
    other tool's.
  - If gunicorn sits behind nginx without X-Forwarded-For, every request
    looks like it came from 127.0.0.1. The script warns when one IP is
    almost everything.
  - Dates come from the log's own timestamps (UTC on most droplets), not
    your local clock. The header prints which offsets were seen.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator


# nginx "combined" and gunicorn's default access format are the same shape.
# search() rather than match() so a syslog prefix or a leading vhost field
# doesn't break the parse.
LINE_RE = re.compile(
    r"(?P<ip>\d{1,3}(?:\.\d{1,3}){3}|[0-9a-fA-F:]{3,39})"
    r"\s+\S+\s+\S+\s+"
    r"\[(?P<ts>[^\]]+)\]\s+"
    r'"(?P<method>[A-Z]+)\s+(?P<path>\S+)[^"]*"\s+'
    r"(?P<status>\d{3})\s+(?P<size>\S+)"
    r'\s+"(?P<referrer>[^"]*)"'
    r'\s+"(?P<ua>[^"]*)"'
)

TS_FORMAT = "%d/%b/%Y:%H:%M:%S %z"

DEFAULT_LOG_GLOBS = (
    "/var/log/nginx/access.log*",
    "/var/log/nginx/justcordelia*.log*",
    "/var/log/gunicorn/access.log*",
    "/var/log/caddy/access.log*",
)

BOT_RE = re.compile(
    r"bot\b|bot/|spider|crawl|slurp|scrapy|wget|curl|python-requests|"
    r"httpx|go-http-client|java/|libwww|okhttp|headless|phantomjs|"
    r"lighthouse|pagespeed|uptime|pingdom|statuscake|monitoring|"
    r"ahrefs|semrush|mj12|dotbot|blexbot|petal|yandex|baidu|sogou|"
    r"facebookexternalhit|whatsapp|telegram|slackbot|discord|embedly|"
    r"gptbot|claudebot|anthropic|ccbot|perplexity|bytespider|amazonbot|"
    r"applebot|duckduckbot|bingpreview|google-read-aloud|feedfetcher",
    re.I,
)

STATIC_RE = re.compile(
    r"\.(?:css|js|mjs|map|png|jpe?g|gif|svg|ico|webp|avif|woff2?|ttf|eot|"
    r"mp4|webm|txt|json|zip|gz)$",
    re.I,
)

# Not page views by any reasonable reading, even when they return HTML.
NOISE_PREFIXES = ("/static/", "/healthz", "/health", "/favicon", "/.well-known/")

SEARCH_ENGINES = (
    ("Bing", re.compile(r"\.bing\.|^bing\.|msn\.com", re.I)),
    ("Google", re.compile(r"google\.", re.I)),
    ("DuckDuckGo", re.compile(r"duckduckgo\.", re.I)),
    ("Yahoo", re.compile(r"yahoo\.", re.I)),
    ("Ecosia", re.compile(r"ecosia\.", re.I)),
    ("Brave", re.compile(r"search\.brave\.", re.I)),
)

SOCIAL_RE = re.compile(
    r"reddit\.|facebook\.|instagram\.|t\.co/|twitter\.|x\.com|linkedin\.|"
    r"pinterest\.|youtube\.|tiktok\.",
    re.I,
)


def open_log(path: Path):
    """Open a log file, transparently handling .gz rotations."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", errors="replace")
    return path.open("r", errors="replace")


def resolve_logs(patterns: list[str]) -> list[Path]:
    """Expand the given paths/globs, or probe the usual locations."""
    globs = patterns or list(DEFAULT_LOG_GLOBS)
    found: list[Path] = []
    for pattern in globs:
        p = Path(pattern)
        if p.exists() and p.is_file():
            found.append(p)
            continue
        # Path.glob needs the pattern split from its anchor.
        anchor = Path(p.anchor) if p.is_absolute() else Path(".")
        relative = str(p.relative_to(p.anchor)) if p.is_absolute() else str(p)
        try:
            found.extend(sorted(x for x in anchor.glob(relative) if x.is_file()))
        except (ValueError, OSError):
            continue
    # Deduplicate, keep order.
    seen: set[Path] = set()
    unique = []
    for path in found:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def iter_records(paths: list[Path], stats: Counter) -> Iterator[dict]:
    """Yield one parsed dict per readable access-log line."""
    for path in paths:
        try:
            handle = open_log(path)
        except OSError as exc:
            print(f"  ! cannot read {path}: {exc}", file=sys.stderr)
            stats["unreadable_files"] += 1
            continue
        with handle:
            for line in handle:
                stats["lines"] += 1
                match = LINE_RE.search(line)
                if not match:
                    stats["unparsed"] += 1
                    continue
                try:
                    when = datetime.strptime(match.group("ts"), TS_FORMAT)
                except ValueError:
                    stats["bad_timestamp"] += 1
                    continue
                stats["parsed"] += 1
                yield {
                    "ip": match.group("ip"),
                    "when": when,
                    "method": match.group("method"),
                    "path": match.group("path"),
                    "status": int(match.group("status")),
                    "referrer": match.group("referrer"),
                    "ua": match.group("ua"),
                }


def is_bot(record: dict) -> bool:
    ua = record["ua"]
    return not ua or ua == "-" or bool(BOT_RE.search(ua))


def is_page_view(record: dict, include_admin: bool) -> bool:
    """A human-meaningful page request, not an asset or a probe."""
    if record["method"] not in ("GET", "HEAD"):
        return False
    if not 200 <= record["status"] < 400:
        return False
    path = record["path"].split("?", 1)[0]
    if STATIC_RE.search(path):
        return False
    if path.startswith(NOISE_PREFIXES):
        return False
    if not include_admin and path.startswith("/admin"):
        return False
    return True


def classify_source(referrer: str, own_host: str) -> str:
    """Bucket a referrer into something you can act on."""
    if not referrer or referrer == "-":
        return "Direct / none"
    if own_host and own_host.lower() in referrer.lower():
        return "Internal"
    for name, pattern in SEARCH_ENGINES:
        if pattern.search(referrer):
            return name
    if SOCIAL_RE.search(referrer):
        return "Social"
    return "Other referral"


def visitor_key(record: dict) -> tuple[str, str]:
    return (record["ip"], record["ua"])


def write_csv(path: str, header: list[str], rows: list[list]) -> None:
    """Write rows to a CSV file, or to stdout when path is '-'.

    Every day in the window gets a row, including zero days — a gap in the
    dates would quietly distort any growth rate calculated from the file.
    """
    if path == "-":
        writer = csv.writer(sys.stdout)
        writer.writerow(header)
        writer.writerows(rows)
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"  wrote {len(rows)} row(s) to {path}", file=sys.stderr)


def day_range(start: datetime, end: datetime) -> list[str]:
    """Every YYYY-MM-DD from start to end inclusive."""
    days = []
    cursor = start
    while cursor <= end:
        days.append(cursor.strftime("%Y-%m-%d"))
        cursor += timedelta(days=1)
    return days


LOCAL_IP_RE = re.compile(
    r"^(?:127\.|10\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.|::1$|fc|fd)",
    re.I,
)


def is_local_address(ip: str) -> bool:
    """True for loopback/private addresses — i.e. the reverse proxy itself."""
    return bool(LOCAL_IP_RE.match(ip))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Count real visitors from web server access logs.",
    )
    parser.add_argument(
        "logs", nargs="*",
        help="Log files or globs. Default: probe the usual nginx/gunicorn paths.",
    )
    parser.add_argument(
        "--days", type=int, default=7,
        help="How many days back to report, counting today (default: 7).",
    )
    parser.add_argument(
        "--host", default="justcordelia.com",
        help="Own hostname, used to label internal referrers.",
    )
    parser.add_argument(
        "--by-source", action="store_true",
        help="Break the most recent day down by referrer source.",
    )
    parser.add_argument(
        "--top-pages", type=int, default=0, metavar="N",
        help="Also list the N busiest pages over the window.",
    )
    parser.add_argument(
        "--bots", action="store_true",
        help="Report bot traffic instead of human traffic.",
    )
    parser.add_argument(
        "--include-admin", action="store_true",
        help="Count /admin requests as page views (excluded by default).",
    )
    parser.add_argument(
        "--csv", metavar="PATH",
        help="Write daily totals as CSV (date,page_views,visitors). "
             "Use - for stdout.",
    )
    parser.add_argument(
        "--csv-by-source", metavar="PATH",
        help="Write the per-source daily split as CSV "
             "(date,source,page_views,visitors). Use - for stdout.",
    )
    args = parser.parse_args()

    # When a CSV goes to stdout, the human-readable report must not — a
    # redirect like `--csv - > daily.csv` would otherwise write the
    # preamble into the CSV file. Send the report to stderr instead.
    csv_to_stdout = "-" in (args.csv, args.csv_by_source)
    out = sys.stderr if csv_to_stdout else sys.stdout

    paths = resolve_logs(args.logs)
    if not paths:
        print("No access logs found.", file=sys.stderr)
        print(
            "Pass a path explicitly, e.g.\n"
            "  python3 scripts/traffic_from_logs.py /var/log/nginx/access.log*",
            file=sys.stderr,
        )
        return 1

    cutoff = datetime.now(timezone.utc) - timedelta(days=max(args.days, 1) - 1)
    cutoff = cutoff.replace(hour=0, minute=0, second=0, microsecond=0)

    stats: Counter = Counter()
    offsets: set[str] = set()
    views_by_day: Counter = Counter()
    visitors_by_day: defaultdict[str, set] = defaultdict(set)
    source_by_day: defaultdict[str, Counter] = defaultdict(Counter)
    source_visitors: defaultdict[str, defaultdict[str, set]] = defaultdict(
        lambda: defaultdict(set))
    pages: Counter = Counter()
    ip_hits: Counter = Counter()

    for record in iter_records(paths, stats):
        if record["when"] < cutoff:
            continue
        offsets.add(record["when"].strftime("%z") or "+0000")
        bot = is_bot(record)
        stats["bot_requests" if bot else "human_requests"] += 1
        if bot != args.bots:
            continue
        if not is_page_view(record, args.include_admin):
            continue

        day = record["when"].strftime("%Y-%m-%d")
        views_by_day[day] += 1
        visitors_by_day[day].add(visitor_key(record))
        ip_hits[record["ip"]] += 1
        pages[record["path"].split("?", 1)[0]] += 1

        source = classify_source(record["referrer"], args.host)
        source_by_day[day][source] += 1
        source_visitors[day][source].add(visitor_key(record))

    label = "BOT" if args.bots else "HUMAN"
    print(f"{label} traffic from {len(paths)} log file(s)", file=out)
    for path in paths:
        print(f"  {path}", file=out)
    print(
        f"  parsed {stats['parsed']:,} of {stats['lines']:,} lines"
        + (f", {stats['unparsed']:,} unrecognised" if stats["unparsed"] else ""),
        file=out,
    )
    if offsets:
        print(f"  log timezone offset(s): {', '.join(sorted(offsets))}", file=out)
    print(file=out)

    # CSV first, so an empty window still produces a complete zero-filled
    # file rather than nothing.
    all_days = day_range(cutoff, datetime.now(timezone.utc))
    if args.csv:
        write_csv(
            args.csv,
            ["date", "page_views", "visitors"],
            [[d, views_by_day.get(d, 0), len(visitors_by_day.get(d, ()))]
             for d in all_days],
        )
    if args.csv_by_source:
        names = sorted({s for day in source_by_day.values() for s in day})
        write_csv(
            args.csv_by_source,
            ["date", "source", "page_views", "visitors"],
            [[d, name,
              source_by_day.get(d, {}).get(name, 0),
              len(source_visitors.get(d, {}).get(name, ()))]
             for d in all_days for name in names],
        )

    if not views_by_day:
        print(f"No {label.lower()} page views in the last {args.days} day(s).", file=out)
        return 0

    print(f"{'Date':<12}{'Page views':>12}{'Visitors':>11}", file=out)
    print("-" * 35, file=out)
    for day in sorted(views_by_day):
        print(f"{day:<12}{views_by_day[day]:>12,}{len(visitors_by_day[day]):>11,}", file=out)
    print("-" * 35, file=out)
    total_visitors = len(set().union(*visitors_by_day.values()))
    print(f"{'Total':<12}{sum(views_by_day.values()):>12,}{total_visitors:>11,}", file=out)
    print("  (visitors are not additive across days — the total is deduplicated)", file=out)
    print(file=out)

    # A single IP dominating means the proxy's own address is being logged
    # instead of the real client, which makes visitor counts meaningless.
    # Deliberately no "more than one IP" guard: total collapse to a single
    # address is the worst case, not a reason to stay quiet.
    if ip_hits:
        top_ip, top_count = ip_hits.most_common(1)[0]
        share = top_count / sum(ip_hits.values())
        if is_local_address(top_ip) or (share > 0.8 and sum(ip_hits.values()) >= 20):
            print(
                f"  ! {share:.0%} of requests come from {top_ip}"
                + (" (a proxy/loopback address)." if is_local_address(top_ip) else ".")
                + "\n    Visitor counts are meaningless until nginx passes"
                "\n    X-Forwarded-For and gunicorn runs with"
                " --forwarded-allow-ips.\n"
            )

    if args.by_source:
        latest = max(views_by_day)
        print(f"Sources for {latest}", file=out)
        print(f"{'Source':<18}{'Page views':>12}{'Visitors':>11}", file=out)
        print("-" * 41, file=out)
        for source, count in source_by_day[latest].most_common():
            visitors = len(source_visitors[latest][source])
            print(f"{source:<18}{count:>12,}{visitors:>11,}", file=out)
        print(file=out)

    if args.top_pages > 0:
        print(f"Top {args.top_pages} pages over the window", file=out)
        width = max((len(p) for p, _ in pages.most_common(args.top_pages)), default=4)
        width = min(width, 70)
        for path, count in pages.most_common(args.top_pages):
            print(f"  {count:>6,}  {path[:width]}", file=out)
        print(file=out)

    other = stats["bot_requests"] if not args.bots else stats["human_requests"]
    other_label = "bot" if not args.bots else "human"
    print(f"Also in window: {other:,} {other_label} requests (use "
          f"{'--bots' if not args.bots else 'no --bots'} to see them).",
          file=out)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        # Piping into head/less closes the pipe early; that isn't an error.
        try:
            sys.stdout.close()
        finally:
            raise SystemExit(0)
