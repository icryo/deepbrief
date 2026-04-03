"""APScheduler — configurable pipeline trigger running inside the web process."""

import logging

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

log = logging.getLogger(__name__)
scheduler = BackgroundScheduler()

JOB_ID = "weekly_run"

# Default: Sunday 22:00 UTC
_DEFAULT_CRON = CronTrigger(day_of_week="sun", hour=22, minute=0)


def _parse_cron_trigger(cron_str: str) -> CronTrigger | None:
    """Parse a 5-field cron string into a CronTrigger.

    Returns None for empty string (manual-only mode).
    Falls back to default schedule on parse errors.
    """
    if not cron_str or not cron_str.strip():
        return None

    parts = cron_str.strip().split()
    if len(parts) != 5:
        log.warning("Invalid cron string '%s' (expected 5 fields) — using default", cron_str)
        return _DEFAULT_CRON

    try:
        return CronTrigger(
            minute=parts[0],
            hour=parts[1],
            day=parts[2],
            month=parts[3],
            day_of_week=parts[4],
        )
    except (ValueError, TypeError) as e:
        log.warning("Failed to parse cron '%s': %s — using default", cron_str, e)
        return _DEFAULT_CRON


def weekly_run():
    """Run all enabled pipelines: aiml → security → github → events → reports."""
    from src.config import is_pipeline_enabled

    log.info("Starting scheduled run ...")

    if is_pipeline_enabled("aiml"):
        try:
            from src.pipelines.aiml import run_aiml_pipeline
            from src.scoring import rescore_top, score_run

            aiml_run_id = run_aiml_pipeline()
            score_run(aiml_run_id, "aiml")
            rescore_top(aiml_run_id, "aiml")

            from src.web.app import _enrich_s2, _generate_report
            _enrich_s2(aiml_run_id, "aiml")
            _generate_report(aiml_run_id, "aiml")
        except Exception:
            log.exception("AI/ML pipeline failed")
    else:
        log.info("AI/ML pipeline disabled — skipping")

    if is_pipeline_enabled("security"):
        try:
            from src.pipelines.security import run_security_pipeline
            from src.scoring import rescore_top, score_run

            sec_run_id = run_security_pipeline()
            score_run(sec_run_id, "security")
            rescore_top(sec_run_id, "security")

            from src.web.app import _enrich_s2, _generate_report
            _enrich_s2(sec_run_id, "security")
            _generate_report(sec_run_id, "security")
        except Exception:
            log.exception("Security pipeline failed")
    else:
        log.info("Security pipeline disabled — skipping")

    if is_pipeline_enabled("github"):
        try:
            from src.pipelines.github import run_github_pipeline
            run_github_pipeline()
        except Exception:
            log.exception("GitHub pipeline failed")
    else:
        log.info("GitHub pipeline disabled — skipping")

    if is_pipeline_enabled("events"):
        try:
            from src.pipelines.events import run_events_pipeline
            run_events_pipeline()
        except Exception:
            log.exception("Events pipeline failed")
    else:
        log.info("Events pipeline disabled — skipping")

    if is_pipeline_enabled("cli_intel"):
        try:
            from src.pipelines.cli_intel import run_cli_intel_pipeline
            run_cli_intel_pipeline()
        except Exception:
            log.exception("CLI intel pipeline failed")
    else:
        log.info("CLI intel pipeline disabled — skipping")

    # Recompute preferences after scoring so adjusted rankings reflect new data
    try:
        from src.preferences import compute_preferences
        from src.db import get_signal_counts
        counts = get_signal_counts()
        if sum(counts.values()) > 0:
            compute_preferences()
            log.info("Preferences recomputed after scheduled run")
    except Exception:
        log.exception("Post-run preference recompute failed")

    log.info("Scheduled run complete")


def start_scheduler():
    """Start the background scheduler with the configured cron job."""
    from src.config import SCHEDULE_CRON

    trigger = _parse_cron_trigger(SCHEDULE_CRON)
    if trigger is None:
        log.info("Schedule set to manual — no automatic jobs")
        scheduler.start()
        return

    scheduler.add_job(
        weekly_run,
        trigger=trigger,
        id=JOB_ID,
        name="Scheduled research pipeline",
        replace_existing=True,
    )
    scheduler.start()
    log.info("Scheduler started with cron: %s", SCHEDULE_CRON)


def reschedule(cron_str: str | None = None):
    """Update the scheduler with a new cron string at runtime.

    If cron_str is None, reads from current config.
    """
    if cron_str is None:
        from src.config import SCHEDULE_CRON
        cron_str = SCHEDULE_CRON

    # Remove existing job if present
    try:
        scheduler.remove_job(JOB_ID)
    except Exception:
        pass  # Job may not exist

    trigger = _parse_cron_trigger(cron_str)
    if trigger is None:
        log.info("Rescheduled to manual — removed automatic job")
        return

    scheduler.add_job(
        weekly_run,
        trigger=trigger,
        id=JOB_ID,
        name="Scheduled research pipeline",
        replace_existing=True,
    )
    log.info("Rescheduled with cron: %s", cron_str)
