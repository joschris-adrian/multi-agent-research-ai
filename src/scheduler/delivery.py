import os
import smtplib
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import httpx


def deliver(report: str, topic: str, delivery_method: str, delivery_target: str):
    """
    Deliver a report via the configured method.
    delivery_method: "email" | "slack" | "discord" | "log"
    delivery_target: email address, Slack webhook URL, or Discord webhook URL
    """
    if delivery_method == "email":
        _send_email(report, topic, delivery_target)
    elif delivery_method in ("slack", "discord"):
        _send_webhook(report, topic, delivery_target, delivery_method)
    elif delivery_method == "log":
        _log(report, topic)
    else:
        print(f"[delivery] unknown method: {delivery_method}")


def _send_email(report: str, topic: str, recipient: str):
    smtp_host = os.environ.get("SMTP_HOST", "")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_password = os.environ.get("SMTP_PASSWORD", "")

    if not all([smtp_host, smtp_user, smtp_password]):
        print("[delivery] SMTP not configured — set SMTP_HOST, SMTP_USER, SMTP_PASSWORD in .env")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Research Report: {topic}"
    msg["From"] = smtp_user
    msg["To"] = recipient
    msg.attach(MIMEText(report, "plain"))

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, recipient, msg.as_string())
        print(f"[delivery] email sent to {recipient}")
    except Exception as e:
        print(f"[delivery] email failed: {e}")


def _send_webhook(report: str, topic: str, url: str, platform: str):
    preview = report[:2000]
    if platform == "slack":
        payload = {"text": f"*Research Report: {topic}*\n\n{preview}"}
    else:
        payload = {"content": f"**Research Report: {topic}**\n\n{preview}"}
    try:
        response = httpx.post(url, json=payload, timeout=15)
        response.raise_for_status()
        print(f"[delivery] {platform} webhook sent")
    except Exception as e:
        print(f"[delivery] {platform} webhook failed: {e}")


def _log(report: str, topic: str):
    print(f"\n[delivery] log — topic: {topic}")
    print(report[:500])