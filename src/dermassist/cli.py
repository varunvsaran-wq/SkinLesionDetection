"""Command-line driver for the DermAssist graph.

Usage:
    dermassist run --image path/to/lesion.jpg [--thread-id ID]
    dermassist resume --thread-id ID --decision approved [--notes "..."]
    dermassist status --thread-id ID

`run` flows the pipeline until the human-review gate, then pauses and exits.
`resume` re-enters the same thread with a reviewer decision; `approved` finalizes,
while `rejected`/`edited` loops back and pauses again.
"""

from __future__ import annotations

import argparse
import json
import sys
from uuid import uuid4

from langgraph.types import Command
from pydantic import BaseModel

from dermassist.compliance import DISCLAIMER, DISCLAIMER_LONG
from dermassist.graph import build_graph
from dermassist.schemas import LesionState

_RULE = "=" * 70


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _collect_interrupts(snapshot) -> list:
    """Pending interrupt payloads from a state snapshot (newest API: task.interrupts)."""
    payloads = []
    for task in getattr(snapshot, "tasks", ()) or ():
        for itr in getattr(task, "interrupts", ()) or ():
            payloads.append(getattr(itr, "value", itr))
    return payloads


def _print_report(report) -> None:
    """Pretty-print a report that may be a dict or a Pydantic model (or None)."""
    if report is None:
        print("(no report yet)")
        return
    if isinstance(report, BaseModel):
        report = report.model_dump()
    print(json.dumps(report, indent=2, ensure_ascii=False))


def _print_snapshot(snapshot, thread_id: str) -> None:
    print(_RULE)
    print(f"thread_id: {thread_id}")
    values = snapshot.values or {}
    pending = snapshot.next or ()

    interrupts = _collect_interrupts(snapshot)
    if interrupts:
        # Graph is paused at the human-review gate.
        print("STATUS: PAUSED — human review required (HARD STOP).")
        print(_RULE)
        for payload in interrupts:
            report = payload.get("report") if isinstance(payload, dict) else None
            print("\nReport awaiting review:")
            _print_report(report)
        print(f"\n⚠️  {DISCLAIMER}")
        print(_RULE)
        print("To resume:")
        print(
            f"  dermassist resume --thread-id {thread_id} "
            f"--decision approved|rejected|edited [--notes \"...\"]"
        )
        print(_RULE)
        return

    if pending:
        print(f"STATUS: RUNNING — pending nodes: {', '.join(pending)}")
        print(_RULE)
        return

    # Terminal: finished.
    print("STATUS: FINALIZED")
    print(_RULE)
    print("\nFinal report:")
    _print_report(values.get("report"))
    print(f"\n⚠️  {DISCLAIMER}")
    print(_RULE)


def cmd_run(args: argparse.Namespace) -> int:
    thread_id = args.thread_id or uuid4().hex[:12]
    graph = build_graph()
    initial = LesionState(image_path=args.image)
    graph.invoke(initial, config=_config(thread_id))
    snapshot = graph.get_state(_config(thread_id))
    _print_snapshot(snapshot, thread_id)
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    graph = build_graph()
    config = _config(args.thread_id)

    snapshot = graph.get_state(config)
    if not _collect_interrupts(snapshot):
        print(
            f"Thread {args.thread_id!r} is not paused at a review gate "
            "(nothing to resume).",
            file=sys.stderr,
        )
        _print_snapshot(snapshot, args.thread_id)
        return 1

    decision = {"status": args.decision, "notes": args.notes}
    graph.invoke(Command(resume=decision), config=config)
    snapshot = graph.get_state(config)
    _print_snapshot(snapshot, args.thread_id)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    graph = build_graph()
    snapshot = graph.get_state(_config(args.thread_id))
    if not snapshot.created_at:
        print(f"No state found for thread {args.thread_id!r}.", file=sys.stderr)
        return 1
    _print_snapshot(snapshot, args.thread_id)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dermassist",
        description=f"DermAssist pipeline. {DISCLAIMER_LONG}",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run the pipeline until the review gate.")
    p_run.add_argument("--image", required=True, help="Path to the dermoscopic image.")
    p_run.add_argument("--thread-id", default=None, help="Optional thread id to use.")
    p_run.set_defaults(func=cmd_run)

    p_resume = sub.add_parser("resume", help="Resume a paused review gate.")
    p_resume.add_argument("--thread-id", required=True)
    p_resume.add_argument(
        "--decision",
        required=True,
        choices=["approved", "rejected", "edited"],
        help="Reviewer decision.",
    )
    p_resume.add_argument("--notes", default=None, help="Optional reviewer notes.")
    p_resume.set_defaults(func=cmd_resume)

    p_status = sub.add_parser("status", help="Show current state for a thread.")
    p_status.add_argument("--thread-id", required=True)
    p_status.set_defaults(func=cmd_status)

    return parser


def main(argv: list[str] | None = None) -> int:
    # The disclaimer/banner use non-ASCII (⚠️). On Windows the default console
    # codec is cp1252 and would crash on print; force UTF-8 where supported.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")

    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
