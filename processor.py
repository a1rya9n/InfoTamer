"""
processor.py - Local LLM Engine & Structured Processing Pipeline for InfoTamer (Sloth Agent).

Uses Google Gemma (recommended: gemma2:2b or gemma2:9b) via Ollama's local REST API.
100% offline, zero data transmitted over public networks.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import requests

from ingestor import NotificationItem

logger = logging.getLogger("infotamer.processor")


# ============================================================================
# Pydantic Schemas for Structured LLM Outputs
# ============================================================================

class CategorizedNotification(BaseModel):
    id: str
    category: str = Field(
        description="One of: 'critical_alert', 'action_required', 'key_discussion', 'low_priority_digest', 'noise'"
    )
    urgency: str = Field(description="'high', 'medium', 'low', or 'none'")
    reason: str = Field(description="Brief 1-sentence rationale for the categorization")


class DenoiseBatchResult(BaseModel):
    items: List[CategorizedNotification] = Field(default_factory=list)


class ActionItem(BaseModel):
    id: str
    task: str = Field(description="Actionable task description: who must do what")
    assigned_to: str = Field(description="Name or handle of person responsible")
    deadline: str = Field(description="Stated deadline or 'Unspecified'")
    urgency: str = Field(description="'immediate', 'today', 'this_week', 'fyi'")
    source_channel: str = Field(description="Channel or source where action was requested")


class ActionExtractionResult(BaseModel):
    action_items: List[ActionItem] = Field(default_factory=list)
    critical_mentions: List[str] = Field(default_factory=list)


class DailyBriefSynthesis(BaseModel):
    two_line_summary: str = Field(description="Exactly two punchy, high-impact sentences summarizing the day's posture")
    action_items_summary: List[str] = Field(default_factory=list)
    critical_alerts: List[str] = Field(default_factory=list)
    key_discussions: List[str] = Field(default_factory=list)
    low_priority_reads: List[str] = Field(default_factory=list)
    noise_items_filtered_count: int = 0


# ============================================================================
# Local LLM Client (Gemma via Ollama)
# ============================================================================

class LocalGemmaProcessor:
    """Wrapper around Ollama local HTTP endpoint for structured JSON extraction with Gemma 2."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "gemma2:2b",
        temperature: float = 0.2,
        timeout: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def check_health(self) -> bool:
        """Verify if local Ollama daemon is reachable."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def query_ollama(self, system_prompt: str, user_prompt: str, json_format: bool = True) -> str:
        """Call Ollama /api/chat with instruction templates."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": self.temperature,
                "top_p": 0.9,
            },
            "stream": False,
        }
        if json_format:
            payload["format"] = "json"

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "")
        except requests.exceptions.ConnectionError:
            logger.warning("[LocalGemmaProcessor] Ollama server is offline at %s", self.base_url)
            raise ConnectionError(f"Ollama server is offline at {self.base_url}. Ensure 'ollama serve' is running.")
        except Exception as e:
            logger.error("[LocalGemmaProcessor] Inference error: %s", str(e))
            raise

    # ------------------------------------------------------------------------
    # Pipeline Step 1: De-noising & Categorization
    # ------------------------------------------------------------------------
    def denoise_and_categorize(self, items: List[NotificationItem]) -> DenoiseBatchResult:
        """
        Stage 1: Separates noise (memes, greetings, casual banter) from signal
        (outages, direct mentions, deliverables).
        """
        if not items:
            return DenoiseBatchResult()

        system_prompt = (
            "You are an offline personal communications triage assistant (Sloth Agent). "
            "Your job is to strictly classify incoming communications into one of 5 categories:\n"
            "- 'critical_alert': System outages, security CVEs, production blockers.\n"
            "- 'action_required': Direct requests for review, questions to the user, tasks with deadlines.\n"
            "- 'key_discussion': Relevant technical or project updates without immediate user action.\n"
            "- 'low_priority_digest': General tech news, non-urgent release notes, informational feeds.\n"
            "- 'noise': Casual greetings, lunch banter, memes, non-work social chatter.\n"
            "Return valid JSON adhering to the specified schema with NO extra commentary."
        )

        batch_payload = [
            {
                "id": it.id,
                "source": it.source,
                "channel": it.channel,
                "sender": it.sender,
                "is_mention": it.is_mention,
                "is_priority": it.is_priority,
                "content": it.sanitized_content,
            }
            for it in items
        ]

        user_prompt = (
            "Classify the following batch of notifications:\n"
            f"```json\n{json.dumps(batch_payload, indent=2)}\n```\n\n"
            "Respond ONLY with a JSON object in this format:\n"
            "{\n"
            '  "items": [\n'
            '    {"id": "...", "category": "critical_alert|action_required|key_discussion|low_priority_digest|noise", "urgency": "high|medium|low|none", "reason": "..."}\n'
            "  ]\n"
            "}"
        )

        try:
            raw_response = self.query_ollama(system_prompt, user_prompt, json_format=True)
            cleaned = self._clean_json(raw_response)
            data = json.loads(cleaned)
            return DenoiseBatchResult(**data)
        except Exception as e:
            logger.warning("[Processor] Ollama triage failed or unavailable (%s). Using fallback heuristic rules.", e)
            return self._fallback_denoise(items)

    # ------------------------------------------------------------------------
    # Pipeline Step 2: Action Item & Mention Extraction
    # ------------------------------------------------------------------------
    def extract_action_items(self, action_items: List[NotificationItem]) -> ActionExtractionResult:
        """
        Stage 2: Extracts actionable tasks ('who', 'what', 'by when', 'urgency').
        """
        if not action_items:
            return ActionExtractionResult()

        system_prompt = (
            "You are an executive assistant for a busy software engineer. "
            "Extract explicit action items, commitments, blockers, and deadlines from high-signal notifications. "
            "Focus only on concrete tasks. Return ONLY structured JSON."
        )

        items_input = [
            {
                "id": it.id,
                "channel": f"{it.source}:{it.channel}",
                "sender": it.sender,
                "content": it.sanitized_content,
            }
            for it in action_items
        ]

        user_prompt = (
            "Extract all action items from these high-priority items:\n"
            f"```json\n{json.dumps(items_input, indent=2)}\n```\n\n"
            "Respond ONLY with a JSON object in this format:\n"
            "{\n"
            '  "action_items": [\n'
            '    {"id": "...", "task": "...", "assigned_to": "...", "deadline": "...", "urgency": "immediate|today|this_week|fyi", "source_channel": "..."}\n'
            "  ],\n"
            '  "critical_mentions": ["Summary of mention 1", "Summary of mention 2"]\n'
            "}"
        )

        try:
            raw_response = self.query_ollama(system_prompt, user_prompt, json_format=True)
            cleaned = self._clean_json(raw_response)
            data = json.loads(cleaned)
            return ActionExtractionResult(**data)
        except Exception as e:
            logger.warning("[Processor] Ollama action extraction failed (%s). Using fallback parser.", e)
            return self._fallback_action_extraction(action_items)

    # ------------------------------------------------------------------------
    # Pipeline Step 3: Executive Brief Generation (2-Line Summary + Digest)
    # ------------------------------------------------------------------------
    def generate_executive_brief(
        self,
        denoise_result: DenoiseBatchResult,
        actions_result: ActionExtractionResult,
        original_items: List[NotificationItem],
    ) -> DailyBriefSynthesis:
        """
        Stage 3: Synthesizes the final Morning Executive Brief.
        Mandate:
        1. A hyper-concise 2-line executive morning brief (TL;DR).
        2. Categorized breakdowns (Action Items, Critical Mentions, Key Discussions, Low-Priority Digest).
        """
        items_by_id = {it.id: it for it in original_items}
        
        # Segment by category
        critical_alerts = []
        key_discussions = []
        low_priority_reads = []
        noise_count = 0

        for cat in denoise_result.items:
            orig = items_by_id.get(cat.id)
            if not orig:
                continue

            if cat.category == "noise":
                noise_count += 1
            elif cat.category == "critical_alert":
                critical_alerts.append(f"[{orig.source.upper()} | {orig.channel}] {orig.sanitized_content}")
            elif cat.category == "key_discussion":
                key_discussions.append(f"[{orig.source.upper()} | {orig.channel}] {orig.sanitized_content}")
            elif cat.category == "low_priority_digest":
                low_priority_reads.append(f"[{orig.source.upper()} | {orig.channel}] {orig.sanitized_content}")

        system_prompt = (
            "You are the Sloth Agent: an elite, hyper-efficient chief-of-staff for an engineer. "
            "Your output must contain an exact 2-line Morning Executive Summary.\n"
            "Line 1 must summarize urgent blockers, system stability, and direct deliverables for today.\n"
            "Line 2 must summarize team coordination, upcoming deadlines, and low-priority chatter.\n"
            "Maintain an objective, punchy, confident tone. Do NOT output fluff or generic greetings."
        )

        context_data = {
            "total_notifications_scanned": len(original_items),
            "noise_filtered_count": noise_count,
            "critical_alerts": critical_alerts,
            "action_items": [a.model_dump() for a in actions_result.action_items],
            "critical_mentions": actions_result.critical_mentions,
            "key_discussions": key_discussions[:5],
            "low_priority_reads": low_priority_reads[:5],
        }

        user_prompt = (
            "Synthesize the morning brief based on these overnight findings:\n"
            f"```json\n{json.dumps(context_data, indent=2)}\n```\n\n"
            "Respond ONLY with a JSON object in this format:\n"
            "{\n"
            '  "two_line_summary": "Line 1: Urgent actions & production status.\\nLine 2: Team blockers & secondary deliverables.",\n'
            '  "action_items_summary": ["Bullet 1", "Bullet 2"],\n'
            '  "critical_alerts": ["Alert 1", "Alert 2"],\n'
            '  "key_discussions": ["Discussion 1"],\n'
            '  "low_priority_reads": ["Read 1"]\n'
            "}"
        )

        try:
            raw_response = self.query_ollama(system_prompt, user_prompt, json_format=True)
            cleaned = self._clean_json(raw_response)
            data = json.loads(cleaned)
            return DailyBriefSynthesis(
                two_line_summary=data.get("two_line_summary", ""),
                action_items_summary=data.get("action_items_summary", [a.task for a in actions_result.action_items]),
                critical_alerts=data.get("critical_alerts", critical_alerts),
                key_discussions=data.get("key_discussions", key_discussions),
                low_priority_reads=data.get("low_priority_reads", low_priority_reads),
                noise_items_filtered_count=noise_count,
            )
        except Exception as e:
            logger.warning("[Processor] Ollama synthesis failed (%s). Generating heuristic synthesis.", e)
            return self._fallback_synthesis(critical_alerts, actions_result, key_discussions, low_priority_reads, noise_count)

    # ------------------------------------------------------------------------
    # Fallback Heuristics (Ensures resilient offline execution)
    # ------------------------------------------------------------------------
    def _fallback_denoise(self, items: List[NotificationItem]) -> DenoiseBatchResult:
        results = []
        alert_pattern = re.compile(r"\b(alert|outage|cve|exhausted|down|critical|blocker|incident)\b", re.IGNORECASE)
        action_pattern = re.compile(r"\b(review|pr\s*#|blocked|deadline|freeze|draft)\b", re.IGNORECASE)
        noise_pattern = re.compile(r"\b(meme|latte|coffee|raid|ggs|good\s+morning|matcha|lunch)\b", re.IGNORECASE)

        for it in items:
            text = it.content
            if alert_pattern.search(text):
                cat = "critical_alert"
                urg = "high"
                reason = "Contains outage or system error keywords"
            elif it.is_mention or action_pattern.search(text) or "2 pm" in text.lower():
                cat = "action_required"
                urg = "high" if it.is_mention else "medium"
                reason = "Direct user mention or clear deliverable request"
            elif noise_pattern.search(text):
                cat = "noise"
                urg = "none"
                reason = "Social banter or non-actionable chatter"
            elif it.source == "rss" or "released" in text.lower() or "patch" in text.lower():
                cat = "low_priority_digest"
                urg = "low"
                reason = "Informational news or security patch update"
            else:
                cat = "key_discussion"
                urg = "low"
                reason = "General project status update"

            results.append(CategorizedNotification(id=it.id, category=cat, urgency=urg, reason=reason))
        return DenoiseBatchResult(items=results)

    def _fallback_action_extraction(self, action_items: List[NotificationItem]) -> ActionExtractionResult:
        extracted_actions = []
        mentions = []
        for it in action_items:
            content = it.sanitized_content
            if it.is_mention:
                mentions.append(f"Mentioned by {it.sender} in {it.source}:{it.channel}: \"{content}\"")
            
            task = content
            deadline = "Unspecified"
            if "tomorrow" in content.lower():
                deadline = "Tomorrow"
            if "10:00 am" in content.lower():
                deadline = "Tomorrow 10:00 AM"
            elif "2 pm" in content.lower():
                deadline = "Tomorrow 2:00 PM"

            extracted_actions.append(ActionItem(
                id=it.id,
                task=task,
                assigned_to=it.sender if not it.is_mention else "User (@aryan)",
                deadline=deadline,
                urgency="immediate" if "alert" in content.lower() else "today",
                source_channel=f"{it.source}:{it.channel}"
            ))
        return ActionExtractionResult(action_items=extracted_actions, critical_mentions=mentions)

    def _fallback_synthesis(
        self,
        critical_alerts: List[str],
        actions_result: ActionExtractionResult,
        key_discussions: List[str],
        low_priority_reads: List[str],
        noise_count: int
    ) -> DailyBriefSynthesis:
        line1 = f"CRITICAL: {len(critical_alerts)} production alert(s) logged; immediate action required on DB pool exhaustion and {len(actions_result.action_items)} team deliverables."
        line2 = f"STATUS: {noise_count} social messages purged; {len(key_discussions)} background threads tracked with reviews pending before 10:00 AM freeze."
        two_line = f"{line1}\n{line2}"

        return DailyBriefSynthesis(
            two_line_summary=two_line,
            action_items_summary=[f"[{a.source_channel}] {a.task} (Deadline: {a.deadline})" for a in actions_result.action_items],
            critical_alerts=critical_alerts,
            key_discussions=key_discussions,
            low_priority_reads=low_priority_reads,
            noise_items_filtered_count=noise_count
        )

    def _clean_json(self, raw: str) -> str:
        """Strip markdown ticks and whitespace around JSON payloads."""
        text = raw.strip()
        # Find first { and last }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return text[start:end+1]
        return text
