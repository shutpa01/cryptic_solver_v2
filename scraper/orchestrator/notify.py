#!/usr/bin/env python3
"""Email notification helper for scrapers.

Uses .env for SMTP config. Graceful no-op if SMTP not configured.

Required .env variables (all optional — if missing, notifications are silently skipped):
    SMTP_HOST=smtp.gmail.com
    SMTP_PORT=587
    SMTP_USER=your_email@gmail.com
    SMTP_PASSWORD=your_app_password
    ALERT_EMAIL=recipient@example.com
"""

import os
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Always load .env from project root, regardless of working directory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / '.env')


def _send_email(subject: str, body: str):
    """Send an email. No-op if SMTP not configured."""
    host = os.getenv('SMTP_HOST')
    port = os.getenv('SMTP_PORT')
    user = os.getenv('SMTP_USER')
    password = os.getenv('SMTP_PASSWORD')
    recipient = os.getenv('ALERT_EMAIL')

    if not all([host, port, user, password, recipient]):
        return

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = user
    msg['To'] = recipient

    try:
        with smtplib.SMTP(host, int(port), timeout=15) as server:
            server.starttls()
            server.login(user, password)
            server.sendmail(user, [recipient], msg.as_string())
    except Exception as e:
        print(f"  [notify] Failed to send email: {e}")


def send_failure_email(scraper_name: str, error_message: str):
    """Send an email alert about a scraper failure."""
    _send_email(
        subject=f"Scraper failure: {scraper_name}",
        body=(
            f"Scraper: {scraper_name}\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"\nError:\n{error_message}\n"
        ),
    )


def send_summary_email(job_name: str, results: dict[str, bool], stats: str = ''):
    """Send a summary email after a scraper run.

    results: dict of {scraper_name: True/False}
    stats: optional stats string to append
    """
    failures = [name for name, ok in results.items() if not ok]
    all_ok = len(failures) == 0

    if all_ok:
        subject = f"{job_name}: all OK"
    else:
        subject = f"{job_name}: {len(failures)} FAILED"

    lines = [
        f"Job: {job_name}",
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        lines.append(f"  {name:20} {status}")

    if stats:
        lines.append("")
        lines.append(stats)

    _send_email(subject, '\n'.join(lines))
