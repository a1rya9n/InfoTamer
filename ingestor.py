"""
ingestor.py - Data Ingestion & Privacy Sanitization Layer for InfoTamer (Sloth Agent).

Responsibilities:
1. PII Sanitization & Masking (emails, phone numbers, secret tokens, IP addresses).
2. Local Ingestion from multiple sources:
   - Realistic mock channel streams (Slack, Discord, WhatsApp, Telegram).
   - Live RSS / Atom Feeds.
   - Local JSON file transcripts.
3. Persistent SQLite storage buffer with automated retention auto-purge.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import sqlite3
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ============================================================================
# Pydantic Models
# ============================================================================

class NotificationItem(BaseModel):
    """Normalized communication record across all incoming channels."""
    id: str
    source: str  # e.g., 'slack', 'discord', 'telegram', 'whatsapp', 'rss'
    channel: str  # e.g., '#eng-alerts', 'Team Core', 'TechCrunch'
    sender: str
    timestamp: datetime.datetime
    content: str
    sanitized_content: str = ""
    is_mention: bool = False
    is_priority: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Privacy Sanitization Layer (Zero-Cloud PII Masking)
# ============================================================================

class PIISanitizer:
    """Pre-processing engine that strips identifying and confidential data before LLM inference."""

    PATTERNS = {
        "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "PHONE": re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        "SECRET_KEY": re.compile(r"(?:api_key|token|bearer|secret|password|auth)[\s:=]+([a-zA-Z0-9_\-]{16,})", re.IGNORECASE),
        "IPV4": re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"),
        "CREDIT_CARD": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    }

    @classmethod
    def sanitize(cls, text: str) -> str:
        """Replace all PII and security tokens with standardized redaction tokens."""
        sanitized = text
        
        # Redact Secrets and API tokens first
        sanitized = cls.PATTERNS["SECRET_KEY"].sub(r"token: [REDACTED_SECRET]", sanitized)
        
        # Redact Personal Identifiable Information
        sanitized = cls.PATTERNS["EMAIL"].sub("[REDACTED_EMAIL]", sanitized)
        sanitized = cls.PATTERNS["PHONE"].sub("[REDACTED_PHONE]", sanitized)
        sanitized = cls.PATTERNS["CREDIT_CARD"].sub("[REDACTED_CARD]", sanitized)
        sanitized = cls.PATTERNS["IPV4"].sub("[REDACTED_IP]", sanitized)
        
        return sanitized


# ============================================================================
# Database Buffer (Local SQLite Store)
# ============================================================================

class LocalNotificationBuffer:
    """Local SQLite repository ensuring zero data leaves the user's filesystem."""

    def __init__(self, db_path: str = "./data/infotamer.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        """Initialize tables with local indexation for fast overnight digestion."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS notifications (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    sender TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sanitized_content TEXT NOT NULL,
                    is_mention INTEGER DEFAULT 0,
                    is_priority INTEGER DEFAULT 0,
                    metadata_json TEXT DEFAULT '{}',
                    processed INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_briefs (
                    id TEXT PRIMARY KEY,
                    brief_date TEXT NOT NULL,
                    summary_2line TEXT NOT NULL,
                    full_brief_md TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_processed ON notifications(processed)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON notifications(timestamp)")
            conn.commit()

    def insert_notifications(self, items: List[NotificationItem]) -> int:
        """Batch insert raw notifications, ignoring duplicates based on hash ID."""
        inserted = 0
        with self._get_connection() as conn:
            for item in items:
                try:
                    conn.execute("""
                        INSERT INTO notifications (
                            id, source, channel, sender, timestamp,
                            content, sanitized_content, is_mention, is_priority,
                            metadata_json, processed, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                    """, (
                        item.id,
                        item.source,
                        item.channel,
                        item.sender,
                        item.timestamp.isoformat(),
                        item.content,
                        item.sanitized_content or PIISanitizer.sanitize(item.content),
                        1 if item.is_mention else 0,
                        1 if item.is_priority else 0,
                        json.dumps(item.metadata),
                        datetime.datetime.now(datetime.timezone.utc).isoformat()
                    ))
                    inserted += 1
                except sqlite3.IntegrityError:
                    # Item already in buffer
                    pass
            conn.commit()
        return inserted

    def get_unprocessed(self, limit: int = 200) -> List[NotificationItem]:
        """Fetch pending notifications to be fed to the overnight LLM pipeline."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM notifications 
                WHERE processed = 0 
                ORDER BY timestamp ASC 
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            
            items = []
            for r in rows:
                items.append(NotificationItem(
                    id=r["id"],
                    source=r["source"],
                    channel=r["channel"],
                    sender=r["sender"],
                    timestamp=datetime.datetime.fromisoformat(r["timestamp"]),
                    content=r["content"],
                    sanitized_content=r["sanitized_content"],
                    is_mention=bool(r["is_mention"]),
                    is_priority=bool(r["is_priority"]),
                    metadata=json.loads(r["metadata_json"] or "{}")
                ))
            return items

    def mark_as_processed(self, ids: List[str]) -> None:
        """Mark items as digested in the morning brief."""
        if not ids:
            return
        with self._get_connection() as conn:
            placeholders = ",".join(["?"] * len(ids))
            conn.execute(f"UPDATE notifications SET processed = 1 WHERE id IN ({placeholders})", ids)
            conn.commit()

    def purge_old_records(self, hours: int = 48) -> int:
        """Auto-purge aged data to prevent unbounded storage and guarantee data privacy."""
        cutoff = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=hours)).isoformat()
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM notifications WHERE timestamp < ?", (cutoff,))
            deleted = cursor.rowcount
            conn.commit()
        return deleted


# ============================================================================
# Ingestion Collector Providers
# ============================================================================

class IngestionManager:
    """Orchestrates collectors from RSS, local files, and simulated message streams."""

    def __init__(self, buffer: LocalNotificationBuffer, user_handles: Optional[List[str]] = None, priority_keywords: Optional[List[str]] = None):
        self.buffer = buffer
        self.user_handles = [h.lower() for h in (user_handles or ["@aryan", "aryan"])]
        self.priority_keywords = [k.lower() for k in (priority_keywords or ["urgent", "outage", "blocker", "pr", "deadline", "review"])]

    def _detect_flags(self, content: str) -> tuple[bool, bool]:
        """Check for user mentions and priority keywords."""
        content_lower = content.lower()
        is_mention = any(h in content_lower for h in self.user_handles)
        is_priority = any(k in content_lower for k in self.priority_keywords)
        return is_mention, is_priority

    def ingest_mock_stream(self) -> List[NotificationItem]:
        """
        Generate realistic multi-channel communication noise + critical signals.
        Simulates: Slack engineering alerts, Discord chat chatter, Telegram releases, WhatsApp family banter.
        """
        now = datetime.datetime.now(datetime.timezone.utc)
        raw_events = [
            # High Signal: Incident
            {
                "source": "slack",
                "channel": "#prod-alerts",
                "sender": "PagerDuty Bot",
                "time_offset_min": -320,
                "content": "ALERT: Database connection pool exhausted on api-worker-3. Latency spiked to 2400ms. Contact @aryan or on-call immediately.",
            },
            # High Signal: Direct mention & Action item
            {
                "source": "slack",
                "channel": "#core-engineering",
                "sender": "Sarah Jenkins",
                "time_offset_min": -290,
                "content": "@aryan could you please review PR #412 (Payment Retry Logic) before tomorrow's 10:00 AM release freeze? We are blocked on your sign-off.",
            },
            # Noise: Office Banter / Memes
            {
                "source": "slack",
                "channel": "#random",
                "sender": "Dave Chen",
                "time_offset_min": -250,
                "content": "anyone want to grab matcha lattes downstairs? also check out this cat meme 🐱 https://imgur.com/example-cat-meme",
            },
            # Medium Signal: Telegram release announcement
            {
                "source": "telegram",
                "channel": "Python Core Updates",
                "sender": "Channel Admin",
                "time_offset_min": -210,
                "content": "Python 3.14.7 security patch released addressing CVE-2026-8812 in socket handling. Upgrades recommended for production clusters.",
            },
            # Noise: Discord gaming & chatter
            {
                "source": "discord",
                "channel": "#gaming-night",
                "sender": "alex99",
                "time_offset_min": -180,
                "content": "GGs on the raid tonight everyone! Same time next Thursday?",
            },
            # High Signal: WhatsApp client/contract deadline
            {
                "source": "whatsapp",
                "channel": "Project Apex Client",
                "sender": "Elena Rostova",
                "time_offset_min": -120,
                "content": "Hi Aryan, just checking if the final architectural draft for the privacy audit will be ready by 2 PM tomorrow? Legal team needs it.",
            },
            # Noise: Good morning banter
            {
                "source": "slack",
                "channel": "#general",
                "sender": "Marcus Brody",
                "time_offset_min": -45,
                "content": "Good morning team! Hope everyone had a great weekend. Coffee machine on 4th floor is working again.",
            },
            # High Signal with Sensitive Data (Tests PII masking):
            {
                "source": "slack",
                "channel": "#ops-private",
                "sender": "DevOps Bot",
                "time_offset_min": -30,
                "content": "Staging server refreshed at 192.168.1.104. Temporary token: bearer_secret_token_991823741829374. Reach out to devops@company.internal or +1-555-019-2834 for access.",
            }
        ]

        items = []
        for ev in raw_events:
            ts = now + datetime.timedelta(minutes=ev["time_offset_min"])
            content = ev["content"]
            is_men, is_prio = self._detect_flags(content)
            
            # Deterministic hash ID
            item_hash = hashlib.sha256(f"{ev['source']}_{ev['channel']}_{ts.isoformat()}_{content}".encode()).hexdigest()[:16]
            
            items.append(NotificationItem(
                id=f"notif_{item_hash}",
                source=ev["source"],
                channel=ev["channel"],
                sender=ev["sender"],
                timestamp=ts,
                content=content,
                sanitized_content=PIISanitizer.sanitize(content),
                is_mention=is_men,
                is_priority=is_prio,
                metadata={"synthetic": True}
            ))

        self.buffer.insert_notifications(items)
        return items

    def ingest_rss_feed(self, feed_url: str, source_label: str = "rss") -> List[NotificationItem]:
        """Fetch and buffer entries from any standard RSS/Atom feed."""
        try:
            import feedparser
        except ImportError:
            print("[WARN] feedparser not installed. Skipping RSS fetch.")
            return []

        parsed = feedparser.parse(feed_url)
        items = []
        for entry in parsed.entries[:10]:
            title = getattr(entry, "title", "Untitled")
            summary = getattr(entry, "summary", "") or getattr(entry, "description", "")
            # Strip simple html tags
            clean_summary = re.sub(r"<[^>]+>", "", summary).strip()
            content = f"{title}\n{clean_summary}"
            
            published = datetime.datetime.now(datetime.timezone.utc)
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                published = datetime.datetime(*entry.published_parsed[:6], tzinfo=datetime.timezone.utc)
            
            is_men, is_prio = self._detect_flags(content)
            item_hash = hashlib.sha256(f"{feed_url}_{title}_{published.isoformat()}".encode()).hexdigest()[:16]
            
            items.append(NotificationItem(
                id=f"rss_{item_hash}",
                source=source_label,
                channel=parsed.feed.get("title", feed_url),
                sender="RSS Bot",
                timestamp=published,
                content=content,
                sanitized_content=PIISanitizer.sanitize(content),
                is_mention=is_men,
                is_priority=is_prio,
                metadata={"link": getattr(entry, "link", "")}
            ))

        self.buffer.insert_notifications(items)
        return items
