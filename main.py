"""
main.py - End-to-End Orchestrator for InfoTamer (The Sloth Agent).

Handles:
1. Environment configuration loading (.env).
2. Command-line interface for manual batch runs, daemon scheduling, and testing.
3. Execution flow: Ingestion -> PII Masking -> Gemma Local LLM Processing -> Report Delivery.
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import time
from typing import List

from dotenv import load_dotenv
from rich.console import Console

from ingestor import LocalNotificationBuffer, IngestionManager, PIISanitizer
from processor import LocalGemmaProcessor
from reporter import BriefReporter

# Load local environment settings
load_dotenv()

console = Console()


def run_pipeline(
    buffer: LocalNotificationBuffer,
    processor: LocalGemmaProcessor,
    reporter: BriefReporter,
    ingest_mock: bool = False,
    rss_url: str = "",
    user_handles: List[str] = None,
    priority_keywords: List[str] = None,
    purge_hours: int = 48,
) -> None:
    """Execute one full overnight digestion and morning brief run."""
    console.print("[bold blue]Starting InfoTamer overnight digestion run...[/bold blue]")

    ingest_mgr = IngestionManager(buffer, user_handles=user_handles, priority_keywords=priority_keywords)

    # 1. Ingestion Step
    if ingest_mock:
        console.print("📥 [cyan]Ingesting simulated overnight communications (Slack, Discord, Telegram, WhatsApp)...[/cyan]")
        mock_items = ingest_mgr.ingest_mock_stream()
        console.print(f"   [green]Buffered {len(mock_items)} mock notifications into local encrypted buffer.[/green]")

    if rss_url:
        console.print(f"📥 [cyan]Ingesting RSS feed: {rss_url}...[/cyan]")
        rss_items = ingest_mgr.ingest_rss_feed(rss_url)
        console.print(f"   [green]Buffered {len(rss_items)} RSS entries into local buffer.[/green]")

    # 2. Retrieve Unprocessed Notifications
    pending = buffer.get_unprocessed(limit=250)
    if not pending:
        console.print("[yellow]No unprocessed notifications found in local buffer. Run with --mock to test.[/yellow]")
        return

    console.print(f"🔍 [bold]Loaded {len(pending)} pending notifications for local LLM triage.[/bold]")

    # 3. Local LLM Processing: Stage 1 (De-noise & Categorize)
    console.print(f"🧠 [magenta]Step 1/3: Classifying signal vs noise using {processor.model}...[/magenta]")
    denoise_result = processor.denoise_and_categorize(pending)

    noise_ids = [c.id for c in denoise_result.items if c.category == "noise"]
    actionable_ids = [c.id for c in denoise_result.items if c.category in ["action_required", "critical_alert"]]
    
    actionable_items = [it for it in pending if it.id in actionable_ids]

    console.print(
        f"   [dim strike]Filtered out {len(noise_ids)} noise/meme messages.[/dim strike] "
        f"[green]Retained {len(actionable_items)} actionable items.[/green]"
    )

    # 4. Local LLM Processing: Stage 2 (Action Item Extraction)
    console.print(f"🧠 [magenta]Step 2/3: Extracting action items and deadlines using {processor.model}...[/magenta]")
    actions_result = processor.extract_action_items(actionable_items)

    # 5. Local LLM Processing: Stage 3 (Executive Brief Synthesis)
    console.print("🧠 [magenta]Step 3/3: Synthesizing 2-line executive brief...[/magenta]")
    synthesis = processor.generate_executive_brief(denoise_result, actions_result, pending)

    # 6. Deliver Brief (Terminal, Markdown File, Desktop Notification)
    reporter.render_terminal_ui(synthesis, actions_result)
    
    md_content = reporter.generate_markdown(synthesis, actions_result)
    saved_path = reporter.save_markdown_report(md_content)
    console.print(f"💾 [bold green]Markdown Brief Saved:[/bold green] [underline]{saved_path}[/underline]")

    # Trigger macOS Notification
    reporter.send_desktop_notification(
        title="Sloth Agent: Morning Brief Ready",
        subtitle=f"{len(actions_result.action_items)} Action Items • {len(synthesis.critical_alerts)} Alerts",
        message=synthesis.two_line_summary.split("\n")[0]
    )

    # 7. Buffer Maintenance
    buffer.mark_as_processed([it.id for it in pending])
    purged = buffer.purge_old_records(hours=purge_hours)
    console.print(f"🧹 [dim]Maintenance: Auto-purged {purged} records older than {purge_hours} hours.[/dim]")
    console.print("[bold green]Digestion run completed successfully.[/bold green]\n")


def start_scheduler_daemon(
    scheduled_time: str,
    buffer: LocalNotificationBuffer,
    processor: LocalGemmaProcessor,
    reporter: BriefReporter,
    **kwargs
) -> None:
    """Run InfoTamer as a background service triggering every morning."""
    from apscheduler.schedulers.blocking import BlockingScheduler

    scheduler = BlockingScheduler()
    hour, minute = map(int, scheduled_time.split(":"))

    def scheduled_job():
        console.print(f"\n[bold yellow]⏰ Scheduled trigger fired at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/bold yellow]")
        run_pipeline(buffer, processor, reporter, **kwargs)

    scheduler.add_job(scheduled_job, "cron", hour=hour, minute=minute)
    console.print(f"⏰ [bold green]InfoTamer Daemon Active.[/bold green] Scheduled to synthesize briefs daily at [bold yellow]{scheduled_time}[/bold yellow].")
    console.print("   (Press Ctrl+C to stop the background runner)")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        console.print("\n[yellow]Daemon stopped by user.[/yellow]")


def main():
    parser = argparse.ArgumentParser(description="InfoTamer (Sloth Agent): Offline-First Notification Tamer")
    parser.add_argument("--mock", action="store_true", help="Ingest simulated Slack, Discord, WhatsApp & Telegram streams")
    parser.add_argument("--rss", type=str, default="", help="Ingest a live RSS feed URL")
    parser.add_argument("--run-now", action="store_true", help="Run the pipeline immediately")
    parser.add_argument("--daemon", action="store_true", help="Run in background daemon mode on schedule")
    parser.add_argument("--test-gemma", action="store_true", help="Test local Gemma LLM connectivity and generation")
    parser.add_argument("--model", type=str, default=os.getenv("LLM_MODEL", "gemma2:2b"), help="Ollama model tag (default: gemma2:2b)")
    parser.add_argument("--base-url", type=str, default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"), help="Ollama server URL")
    parser.add_argument("--db-path", type=str, default=os.getenv("DATABASE_PATH", "./data/infotamer.db"), help="SQLite database path")
    parser.add_argument("--output-dir", type=str, default=os.getenv("OUTPUT_DIR", "./briefs"), help="Daily brief output folder")

    args = parser.parse_args()

    # Initialize components
    buffer = LocalNotificationBuffer(db_path=args.db_path)
    processor = LocalGemmaProcessor(base_url=args.base_url, model=args.model)
    reporter = BriefReporter(output_dir=args.output_dir)

    user_handles = [h.strip() for h in os.getenv("USER_HANDLES", "@aryan,aryan").split(",") if h.strip()]
    priority_keywords = [k.strip() for k in os.getenv("PRIORITY_KEYWORDS", "urgent,outage,critical,pr,review,deadline").split(",") if k.strip()]
    scheduled_time = os.getenv("SCHEDULED_TIME", "06:00")
    purge_hours = int(os.getenv("PURGE_AFTER_HOURS", "48"))

    # Test mode
    if args.test_gemma:
        console.print(f"[bold cyan]Testing local Gemma LLM connection to {args.base_url} with model '{args.model}'...[/bold cyan]")
        if not processor.check_health():
            console.print(f"[bold red]❌ Failed to connect to Ollama at {args.base_url}.[/bold red]")
            console.print("Please make sure Ollama is installed and running (`ollama serve`).")
            sys.exit(1)
        
        console.print("[green]✔ Ollama server is reachable![/green]")
        console.print(f"Generating test completion with '{args.model}'...")
        try:
            res = processor.query_ollama(
                system_prompt="You are a helpful assistant.",
                user_prompt="Explain in 1 sentence why running AI models locally protects user privacy.",
                json_format=False
            )
            console.print(f"\n[bold green]Response from Gemma:[/bold green]\n{res}\n")
            console.print("[bold green]✔ Gemma integration test passed successfully![/bold green]")
        except Exception as e:
            console.print(f"[bold red]❌ Inference failed: {e}[/bold red]")
            sys.exit(1)
        return

    # Daemon mode
    if args.daemon:
        start_scheduler_daemon(
            scheduled_time=scheduled_time,
            buffer=buffer,
            processor=processor,
            reporter=reporter,
            ingest_mock=args.mock,
            rss_url=args.rss,
            user_handles=user_handles,
            priority_keywords=priority_keywords,
            purge_hours=purge_hours,
        )
        return

    # Default to run-now if mock or rss provided or explicitly requested
    if args.run_now or args.mock or args.rss:
        run_pipeline(
            buffer=buffer,
            processor=processor,
            reporter=reporter,
            ingest_mock=args.mock,
            rss_url=args.rss,
            user_handles=user_handles,
            priority_keywords=priority_keywords,
            purge_hours=purge_hours,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
