"""
Urban-GenX | Remote Monitoring
Sends push notifications to ntfy.sh topic: vibhor_urban_genx
Usage: from src.utils.notifier import notify
       notify("Epoch 10/50 | D_Loss: 0.34 | G_Loss: 2.1")
"""

import requests
import traceback

NTFY_TOPIC = "vibhor_urban_genx"
NTFY_URL   = f"https://ntfy.sh/{NTFY_TOPIC}"

def notify(message: str, title: str = "Urban-GenX", priority: str = "default", tags: list = None):
    """
    Send push notification to mobile via ntfy.sh.
    Priority: min / low / default / high / urgent
    Tags:     emoji shortcodes, e.g. ["white_check_mark"] 
    """
    headers = {
        "Title":    title,
        "Priority": priority,
    }
    if tags:
        headers["Tags"] = ",".join(tags)
    try:
        r = requests.post(NTFY_URL, data=message.encode('utf-8'), headers=headers, timeout=5)
        r.raise_for_status()
    except Exception:
        # Never let a notification failure crash training
        pass

# ─── Convenience wrappers ─────────────────────────────────────────────────────
def notify_epoch(epoch, total, d_loss, g_loss):
    notify(
        f"Epoch {epoch}/{total} | D: {d_loss:.4f} | G: {g_loss:.4f}",
        title="🏙️ Vision Training",
        tags=["chart_with_upwards_trend"]
    )

def notify_crash_save(epoch, path):
    notify(
        f"💾 Checkpoint saved → Epoch {epoch} | {path}",
        title="Urban-GenX Checkpoint",
        priority="high",
        tags=["floppy_disk"]
    )

def notify_training_complete(total_epochs, final_g_loss):
    notify(
        f"✅ Training complete! {total_epochs} epochs | Final G_Loss: {final_g_loss:.4f}",
        title="Urban-GenX DONE",
        priority="high",
        tags=["white_check_mark", "tada"]
    )

def notify_error(error_msg: str):
    notify(
        f"❌ ERROR: {error_msg[:200]}",
        title="Urban-GenX CRASH",
        priority="urgent",
        tags=["rotating_light", "sos"]
    )
