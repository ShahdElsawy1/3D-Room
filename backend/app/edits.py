"""Week 2 task specs: for every edit task type, what the worker downloads, which stage it runs with which
arguments, and what it uploads. The backend is the single source of this contract, so the worker stays generic.

Stages (scripts the worker runs, with --project-dir <root>/<project_id> --backend-url --task-id appended):
    segment       Person A   ML_Work/stages/w2_segment.py --edit-id E (--frame F --clicks JSON | --add-frames JSON)
    track         Person A   ML_Work/stages/w2_track.py   --edit-id E
    edit_preview  Person B   backend/editing/run_edit.py preview --edit-id E --prompt P --seeds 0,1,2 ...
    edit_apply    Person B   backend/editing/run_edit.py frames  --edit-id E --prompt P --chosen-seed S ...
    rebuild       Person C/A ML_Work/stages/w2_rebuild.py --edit-id E --mode finetune|scratch
Each writes into <project>/edits/E/ (layout: docs/W2_INTERFACES_PROPOSAL.md).
"""
from __future__ import annotations

import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from .artifacts import REQUIRED, kind_spec
from .models import Edit, Task, TaskStatus

ORIGINAL, EDIT = "original", "edit"

# (kind, version, required). version EDIT = this task's edit id.
INPUTS: dict[str, list[tuple[str, str, bool]]] = {
    "segment": [("frames", ORIGINAL, True), ("cameras", ORIGINAL, False), ("masks", EDIT, False)],
    "track": [("frames", ORIGINAL, True), ("scene", ORIGINAL, True), ("cameras", ORIGINAL, False),
              ("masks", EDIT, True)],
    "edit_preview": [("frames", ORIGINAL, True), ("masks", EDIT, True)],
    "edit_apply": [("frames", ORIGINAL, True), ("masks", EDIT, True), ("candidates", EDIT, True)],
    "rebuild": [("frames", ORIGINAL, True), ("scene", ORIGINAL, True), ("cameras", ORIGINAL, False),
                ("edited", EDIT, True), ("masks", EDIT, False), ("gs", ORIGINAL, False)],
}
OPTIONAL_OUTPUTS = {"rebuild": ["metrics", "ply"], "edit_preview": [], "edit_apply": [], "segment": [], "track": []}
STAGE = {"segment": "w2_segment", "track": "w2_track", "rebuild": "w2_rebuild",
         "edit_preview": "run_edit", "edit_apply": "run_edit"}

# status of an edit = what its latest task is doing / has produced (proposal §2)
_ACTIVE_STATUS = {"segment": "segmenting", "track": "tracking", "edit_preview": "previewing",
                  "edit_apply": "editing", "rebuild": "rebuilding"}
_DONE_STATUS = {"segment": "mask_ready", "track": "tracked", "edit_preview": "preview_ready",
                "edit_apply": "edited", "rebuild": "done"}


def is_edit_task(task_type: str) -> bool:
    return task_type in INPUTS


def _ref(kind: str, version: str, required: bool) -> dict:
    k = kind_spec(kind, version)
    return {"kind": kind, "version": version, "path": k.path, "is_dir": k.is_dir, "required": required}


def inputs_for(task: Task) -> list[dict]:
    eid = task.params["edit_id"]
    return [_ref(k, eid if v == EDIT else ORIGINAL, req) for k, v, req in INPUTS[task.type]]


def outputs_for(task: Task) -> list[dict]:
    eid = task.params["edit_id"]
    return ([_ref(k, eid, True) for k in REQUIRED[task.type]] +
            [_ref(k, eid, False) for k in OPTIONAL_OUTPUTS[task.type]])


def stage_args(task: Task) -> list[str]:
    """Arguments for the stage script, built only from the task's params (frozen when the task was created)."""
    p = task.params
    eid = p["edit_id"]
    if task.type == "segment":
        if p.get("add_frames"):
            return ["--edit-id", eid, "--add-frames", json.dumps(p["add_frames"])]
        return ["--edit-id", eid, "--frame", p["frame"], "--clicks", json.dumps(p["clicks"])]
    if task.type == "track":
        return ["--edit-id", eid]
    if task.type == "rebuild":
        return ["--edit-id", eid, "--mode", p["mode"]]
    common = ["--edit-id", eid, "--prompt", p["prompt"], "--model", p["model"], "--precision", p["precision"],
              "--placement", p["placement"], "--long-side", str(p["long_side"])]
    if p.get("allow_fp16"):
        common.append("--allow-fp16")
    if task.type == "edit_preview":
        return ["preview", *common, "--seeds", ",".join(map(str, p["seeds"])), "--ref-frame", p["ref_frame"]]
    out = ["frames", *common, "--chosen-seed", str(p["chosen_seed"])]
    if p.get("exclude"):
        out += ["--exclude", ",".join(p["exclude"])]
    if p.get("with_previous"):
        out.append("--with-previous")
    return out


def edit_tasks(db: Session, edit: Edit) -> list[Task]:
    rows = db.scalars(select(Task).where(Task.project_id == edit.project_id).order_by(Task.created_at)).all()
    return [t for t in rows if (t.params or {}).get("edit_id") == edit.id]


def edit_status(tasks: list[Task]) -> str:
    if not tasks:
        return "created"
    t = tasks[-1]
    if t.status in (TaskStatus.pending.value, TaskStatus.running.value):
        return _ACTIVE_STATUS[t.type]
    if t.status == TaskStatus.done.value:
        return _DONE_STATUS[t.type]
    return f"{t.type}_{t.status}"      # e.g. edit_apply_failed, track_cancelled
