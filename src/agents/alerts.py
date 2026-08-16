# src/agent/alerts.py

import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
from pathlib import Path


def send_alert(
    anomaly,
    clip_path: str = None,
    smtp_host: str = "smtp.gmail.com",
    smtp_port: int = 587,
    sender_email: str = None,
    sender_password: str = None,
    recipient_email: str = None,
):
    """
    Send an email alert with the anomaly details and optionally a video clip.
    
    For Gmail, use an App Password:
    https://myaccount.google.com/apppasswords
    
    Set these environment variables:
        ALERT_SENDER_EMAIL
        ALERT_SENDER_PASSWORD  
        ALERT_RECIPIENT_EMAIL
    """
    sender_email = sender_email or os.environ.get("ALERT_SENDER_EMAIL")
    sender_password = sender_password or os.environ.get("ALERT_SENDER_PASSWORD")
    recipient_email = recipient_email or os.environ.get("ALERT_RECIPIENT_EMAIL")

    if not all([sender_email, sender_password, recipient_email]):
        print("  [alert] Email credentials not set — skipping email alert.")
        print("  Set ALERT_SENDER_EMAIL, ALERT_SENDER_PASSWORD, ALERT_RECIPIENT_EMAIL")
        return False

    # build email
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = f"[VideoLens Alert] Anomaly Detected — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    # body
    timestamp = datetime.fromtimestamp(anomaly.timestamp / 1e6).strftime("%Y-%m-%d %H:%M:%S")
    body = f"""
VideoLens Anomaly Alert
=======================
Time:           {timestamp}
Scene:          {anomaly.scene}
Frame:          {anomaly.frame_idx}
Anomaly Score:  {anomaly.anomaly_score}
Objects:        {', '.join(set(anomaly.detected_classes)) or 'None'}
Reasons:
{chr(10).join(f'  - {r}' for r in anomaly.reasons)}

Image: {anomaly.image_path}
"""
    msg.attach(MIMEText(body, "plain"))

    # attach frame image
    if anomaly.image_path and Path(anomaly.image_path).exists():
        with open(anomaly.image_path, "rb") as f:
            img = MIMEImage(f.read())
            img.add_header("Content-Disposition", "attachment",
                          filename=Path(anomaly.image_path).name)
            msg.attach(img)

    # attach video clip if available
    if clip_path and Path(clip_path).exists():
        with open(clip_path, "rb") as f:
            clip = MIMEBase("video", "mp4")
            clip.set_payload(f.read())
            encoders.encode_base64(clip)
            clip.add_header("Content-Disposition", "attachment",
                           filename=Path(clip_path).name)
            msg.attach(clip)

    # send
    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
        print(f"  [alert] Email sent to {recipient_email}")
        return True
    except Exception as e:
        print(f"  [alert] Failed to send email: {e}")
        return False


def log_alert(anomaly, log_path: str = "agent_log.json"):
    """Log anomaly to a JSON file regardless of email status."""
    import json
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "frame_idx": anomaly.frame_idx,
        "scene": anomaly.scene,
        "image_path": anomaly.image_path,
        "anomaly_score": anomaly.anomaly_score,
        "detected_classes": anomaly.detected_classes,
        "reasons": anomaly.reasons,
        "triggered": anomaly.triggered,
    }

    logs = []
    if Path(log_path).exists():
        with open(log_path) as f:
            logs = json.load(f)
    
    logs.append(log_entry)
    
    with open(log_path, "w") as f:
        json.dump(logs, f, indent=2)
    
    print(f"  [alert] Logged to {log_path}")