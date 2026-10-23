"""
reporter.py - Executive Morning Brief Generator for InfoTamer (Sloth Agent).

Produces:
1. Hyper-concise 2-line Morning Executive Summary.
2. Structured Markdown Daily Digest.
3. Rich Terminal Interface (ANSI tables, colored panels).
4. macOS Desktop Notification (via native osascript).
"""

from __future__ import annotations

import datetime
import os
import subprocess
import sys
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from processor import DailyBriefSynthesis, ActionExtractionResult


class BriefReporter:
    """Formats and dispatches the daily morning briefing."""

    def __init__(self, output_dir: str = "./briefs"):
        self.output_dir = output_dir
        self.console = Console()
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_markdown(
        self,
        synthesis: DailyBriefSynthesis,
        actions: ActionExtractionResult,
        run_date: Optional[datetime.date] = None,
    ) -> str:
        """Construct the persistent Markdown report."""
        date_str = (run_date or datetime.date.today()).strftime("%A, %B %d, %Y")
        lines = synthesis.two_line_summary.strip().split("\n")
        line1 = lines[0] if len(lines) > 0 else "No urgent blockers reported."
        line2 = lines[1] if len(lines) > 1 else "All systems operating within normal parameters."

        md = f"""# 🦥 The Sloth Agent: Morning Executive Brief
**Date:** {date_str}  
**Confidentiality:** 100% Offline Local Inference (No Data Exfiltration)

---

## ⚡ 2-Line Executive Summary (TL;DR)
> **1.** {line1}  
> **2.** {line2}

---

## 🚨 Critical Alerts & Blockers
"""
        if synthesis.critical_alerts:
            for alert in synthesis.critical_alerts:
                md += f"- ⚠️ **{alert}**\n"
        else:
            md += "- *No active system outages or blockers reported overnight.*\n"

        md += "\n## ✅ Action Items (Immediate & Due Today)\n"
        if actions.action_items:
            for act in actions.action_items:
                badge = "🔴" if act.urgency in ["immediate", "high"] else "🟡"
                md += f"- {badge} **[{act.urgency.upper()}]** {act.task}\n"
                md += f"  - *Assigned to:* `{act.assigned_to}` | *Deadline:* **{act.deadline}** | *Source:* `{act.source_channel}`\n"
        else:
            md += "- *No explicit tasks pending your immediate action.*\n"

        md += "\n## 💬 Critical Mentions & Direct pings\n"
        if actions.critical_mentions:
            for mention in actions.critical_mentions:
                md += f"- 📌 {mention}\n"
        else:
            md += "- *No unresolved direct mentions logged.*\n"

        md += "\n## 📋 Key Discussions & Team Updates\n"
        if synthesis.key_discussions:
            for disc in synthesis.key_discussions:
                md += f"- 💬 {disc}\n"
        else:
            md += "- *No major engineering discussions logged overnight.*\n"

        md += "\n## 📰 Low-Priority Digest (Tech Feeds & News)\n"
        if synthesis.low_priority_reads:
            for item in synthesis.low_priority_reads:
                md += f"- 📰 {item}\n"
        else:
            md += "- *No external feed items collected.*\n"

        md += f"""
---
*Privacy Metrics: Automatically filtered & purged **{synthesis.noise_items_filtered_count}** noise/meme/banter messages overnight. Zero bytes sent to cloud.*
"""
        return md

    def save_markdown_report(self, markdown_content: str, run_date: Optional[datetime.date] = None) -> str:
        """Write the daily brief to a permanent, localized markdown file."""
        date_stamp = (run_date or datetime.date.today()).strftime("%Y-%m-%d")
        file_path = os.path.join(self.output_dir, f"morning_brief_{date_stamp}.md")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        return file_path

    def render_terminal_ui(self, synthesis: DailyBriefSynthesis, actions: ActionExtractionResult) -> None:
        """Render a rich, high-visibility terminal dashboard."""
        self.console.print()
        
        # Header Banner
        header = Text("🦥 THE NOTIFICATION & INFO OVERLOAD TAMER (SLOTH AGENT)", style="bold cyan")
        sub = Text("Local Hardware Execution • 100% Privacy Guarantee • Overnight Brief", style="italic dim")
        self.console.print(Panel(Text.assemble(header, "\n", sub), border_style="cyan"))

        # 2-Line Summary
        summary_lines = synthesis.two_line_summary.strip().split("\n")
        summary_panel = Panel(
            "\n".join([f"[bold yellow]•[/bold yellow] {l}" for l in summary_lines]),
            title="[bold green]☀️ MORNING 2-LINE EXECUTIVE SUMMARY[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
        self.console.print(summary_panel)

        # Action Items Table
        if actions.action_items:
            table = Table(title="⚡ Action Items & Blockers", border_style="magenta", show_lines=True)
            table.add_column("Urgency", style="bold red", width=12)
            table.add_column("Task", style="white", min_width=30)
            table.add_column("Deadline", style="yellow", width=18)
            table.add_column("Source Channel", style="cyan", width=22)

            for act in actions.action_items:
                urg_style = "bold red" if act.urgency in ["immediate", "high"] else "yellow"
                table.add_row(
                    f"[{urg_style}]{act.urgency.upper()}[/{urg_style}]",
                    act.task,
                    act.deadline,
                    act.source_channel,
                )
            self.console.print(table)

        # Stats Panel
        stats_text = (
            f"[bold]Filtered Noise Messages:[/bold] [dim strike]{synthesis.noise_items_filtered_count} discarded[/dim strike]\n"
            f"[bold]Action Items Extracted:[/bold] [green]{len(actions.action_items)}[/green]\n"
            f"[bold]Critical Alerts:[/bold] [red]{len(synthesis.critical_alerts)}[/red]\n"
            f"[bold]Privacy Status:[/bold] [bold green]Local Inference Only (Zero Network Leaks)[/bold green]"
        )
        self.console.print(Panel(stats_text, title="📊 Overnight Ingestion Metrics", border_style="blue"))
        self.console.print()

    def send_desktop_notification(self, title: str, subtitle: str, message: str) -> None:
        """Send a native macOS notification using osascript if running on Mac."""
        if sys.platform != "darwin":
            return
        
        # Clean text for AppleScript
        clean_title = title.replace('"', '\\"')
        clean_sub = subtitle.replace('"', '\\"')
        clean_msg = message.replace('"', '\\"').replace("\n", " ")[:120]

        script = f'display notification "{clean_msg}" with title "{clean_title}" subtitle "{clean_sub}" sound name "Glass"'
        try:
            subprocess.run(["osascript", "-e", script], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
