"""Sends the finished manifesto recording by email. SMTP credentials come
from a .env file in the project root — never committed (see .gitignore),
copy .env.example and fill it in. Not named email.py to avoid any
confusion with the stdlib `email` package this module itself imports from.
"""

from __future__ import annotations

import os
import re
import smtplib
from email.message import EmailMessage
from pathlib import Path

from dotenv import load_dotenv

from . import config

load_dotenv(config.PROJECT_ROOT / ".env")

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

DEFAULT_SUBJECT = "Your Solarpunk Manifesto, in your own voice"
DEFAULT_BODY = (
    "Thank you for lending your voice to the Solarpunk Manifesto.\n\n"
    "Attached is the full manifesto, read entirely in your own cloned voice.\n\n"
    "— solarpunk@udk.ai"
)


class MailerNotConfigured(RuntimeError):
    pass


def is_valid_email(address: str) -> bool:
    return bool(_EMAIL_RE.match((address or "").strip()))


def _smtp_settings() -> tuple[str, int, str, str, str]:
    host = os.environ.get("SMTP_HOST")
    port = os.environ.get("SMTP_PORT")
    username = os.environ.get("SMTP_USERNAME")
    password = os.environ.get("SMTP_PASSWORD")
    from_addr = os.environ.get("SMTP_FROM", "solarpunk@udk.ai")
    if not all([host, port, username, password]):
        raise MailerNotConfigured(
            "SMTP is not configured. Copy .env.example to .env at the project root and fill in "
            "SMTP_HOST, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD (and optionally SMTP_FROM)."
        )
    return host, int(port), username, password, from_addr


def send_manifesto_email(
    to_addr: str,
    attachment_path: Path,
    subject: str = DEFAULT_SUBJECT,
    body: str = DEFAULT_BODY,
) -> None:
    if not is_valid_email(to_addr):
        raise ValueError(f"not a valid email address: {to_addr!r}")

    host, port, username, password, from_addr = _smtp_settings()

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg.set_content(body)

    data = Path(attachment_path).read_bytes()
    msg.add_attachment(data, maintype="audio", subtype="mpeg", filename=Path(attachment_path).name)

    if port == 465:
        with smtplib.SMTP_SSL(host, port, timeout=30) as smtp:
            smtp.login(username, password)
            smtp.send_message(msg)
    else:
        with smtplib.SMTP(host, port, timeout=30) as smtp:
            smtp.starttls()
            smtp.login(username, password)
            smtp.send_message(msg)
