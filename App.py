# new code
import streamlit as st
import pandas as pd
import itertools
import io
import altair as alt
from streamlit import column_config
import math
import os

# DB helpers (Postgres via pg8000) — create tables, read, write
from db import ensure_schema, load_all, save_module_meta, save_cas, save_schedule

# Streamlit page layout
st.set_page_config(layout="wide")

# Small CSS tweak for the “caption-lg” utility class
st.markdown("""
<style>
  .caption-lg{
    font-size: 1.10rem;   /* tweak as you like */
    line-height: 1.35;
    margin: .2rem 0 .7rem 0;
    color: inherit;       /* respects light/dark theme */
    opacity: .95;
  }
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────────────────
# 1) INITIAL LOAD FROM DATABASE
# Ensure tables exist; then load persisted meta, CA maps and schedules into memory.
# ───────────────────────────────────────────────────────────────────────────────
ensure_schema()
modules_meta, cas_map_db, schedules_db = load_all()

# ───────────────────────────────────────────────────────────────────────────────
# 2) SESSION STATE SETUP
# Keep the app state across reruns (navigation step, data copies, selections, etc.)
# ───────────────────────────────────────────────────────────────────────────────
if 'step'     not in st.session_state: st.session_state.step     = 3   # default landing: Results
if 'modules'  not in st.session_state: st.session_state.modules  = schedules_db.copy()
if 'baseline' not in st.session_state: st.session_state.baseline = schedules_db.copy()
if 'meta'     not in st.session_state: st.session_state.meta     = modules_meta.copy()
if 'ca_map'   not in st.session_state: st.session_state.ca_map   = cas_map_db.copy()
if 'ca_names' not in st.session_state: st.session_state.ca_names = {}   # per-module CA display names
if 'weeks'    not in st.session_state:
    st.session_state.weeks = [f"week {i}" for i in range(1,16)]
if 'selected' not in st.session_state:
    st.session_state.selected = None                    # which module is being edited
if 'selected_modules' not in st.session_state:
    st.session_state.selected_modules = []              # (unused here, but reserved)         

if 'selected_ca_map' not in st.session_state:
    st.session_state.selected_ca_map = {}               # (unused here, but reserved)        
if 'heatmap_modules' not in st.session_state:
    st.session_state.heatmap_modules = None             # selection for heatmap view
if 'last_applied_moves' not in st.session_state:
    # record of last “Shift now”: {(module, ca_index): (old_deadline, new_deadline)}
    st.session_state.last_applied_moves = {}

# Support undo of the last applied move set
if 'undo_payload' not in st.session_state:
    st.session_state.undo_payload = None   # {(module, ca_index): old_deadline}
if 'undo_caption' not in st.session_state:
    st.session_state.undo_caption = ""     # human-readable summary of last move set

# ───────────────────────────────────────────────────────────────────────────────
# 3) SIDEBAR NAVIGATION
# Radio switches the app "step" between list, setup, results and heatmaps.
# ───────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Navigate")
    nav_items = ["Results", "Modules", "Module setup", "Heatmaps (all pairs)"]

    # Select default based on current step
    default_choice = (
        "Modules"              if st.session_state.step == 0 else
        "Module setup"         if st.session_state.step == 1 else
        "Heatmaps (all pairs)" if st.session_state.step == "HEATMAP" else
        "Results"
    )

    choice = st.radio(
        "Go to",
        nav_items,
        index=nav_items.index(default_choice),
        label_visibility="collapsed",
        key="nav_choice",
    )

# Map chosen nav to internal step value
if choice == "Results":
    st.session_state.step = 3
elif choice == "Modules":
    st.session_state.step = 0
elif choice == "Module setup":
    st.session_state.step = 1
elif choice == "Heatmaps (all pairs)":
    st.session_state.heatmap_modules = "__ALL__"
    st.session_state.step = "HEATMAP"

# ───────────────────────────────────────────────────────────────────────────────
# 4) HELPER FUNCTIONS
# Colour bands, statistics, transitions between steps, CA table helpers, etc.
# ───────────────────────────────────────────────────────────────────────────────
def colour_for_cv(cv: float) -> str:
    """
    Map a CV-like metric to colour bands used across the UI.
    Returned string names are later mapped to hex colours.
    """
    if cv < 30:         return "green"
    elif cv < 40:       return "yellow"
    elif cv < 50:       return "orange"
    elif cv < 65:       return "red"
    else:               return "black"

def normalize_cv(ser):
    """
    Normalize coefficient of variation: CV / (1 + CV).
    Used if you want a bounded version; not heavily used in this file.
    """
    raw = ser.std(ddof=0) / ser.mean()
    return raw / (1 + raw)

def go_next(): st.session_state.step += 1
def go_prev(): st.session_state.step -= 1
def go_home():
    """Return to Modules list and clear any active selection."""
    st.session_state.step = 0
    st.session_state.selected = None

def ca_df(mod):
    """
    Build a small dataframe of CA rows for a given module.
    Filters to deadlines within weeks 1..12 (teaching weeks).
    """
    cas = st.session_state.ca_map.get(mod, [])
    name_map = st.session_state.ca_names.get(mod, {})  # {idx: name}
    rows = [
        {
            "CA#": idx,
            "Name": name_map.get(idx, f"CA #{idx}"),
            "Release": rl,
            "Deadline": dl,
            "Weight%": round(wt, 1),
        }
        for (idx, wt, dl, rl) in cas
        if 1 <= dl <= 12
    ]
    return pd.DataFrame(rows)

def deadline_set(mod):
    """Return a set of deadline weeks (1..12) used by the module’s CAs."""
    cas = st.session_state.ca_map.get(mod, [])
    return {dl for (_idx, _wt, dl, _rl) in cas if 1 <= dl <= 12}

# ---------- Heatmap & Calendar helpers ----------
def build_calendar_df(weeks_labels=None, overrides=None, moves=None):
    """
    Build a spreadsheet-like calendar:
      - Rows are modules.
      - Columns are weeks 1..15.
      - Cells list CA name and weight at the CA *deadline* week.
    Supports “preview” overrides (yellow dashed) and “applied” moves (blue boxes).
    """
    weeks_labels = weeks_labels or st.session_state.weeks
    overrides = overrides or st.session_state.get("preview_overrides", {})
    moves = moves or st.session_state.get("last_applied_moves", {})
    n_weeks = len(weeks_labels)
    table = {mod: [""] * n_weeks for mod in st.session_state.meta.keys()}

    for mod, cas_list in st.session_state.ca_map.items():
        name_map = st.session_state.ca_names.get(mod, {})
        for (idx, wt, dl, rl) in cas_list:
            if 1 <= dl <= 12:
                final_dl = overrides.get((mod, idx), dl)   # if previewing, move cell label
                moved_pair = moves.get((mod, idx))         # annotate last-applied shift

                label = name_map.get(idx, f"CA #{idx}")
                try:
                    pct = f"({float(wt):.0f}%)" if wt not in (None, 0) else ""
                except Exception:
                    pct = f"({wt}%)" if wt not in (None, 0) else ""
                base_text = f"{label} {pct}".strip()

                # If the last-applied move is still reflected in the data, mark old cell and new cell
                if moved_pair:
                    old_dl, new_dl = moved_pair
                    if dl == new_dl:
                        # show a ghost label in the old position
                        old_idx = old_dl - 1
                        ghost = f"[moved_from] {base_text} → {new_dl}"
                        prev_old = table[mod][old_idx]
                        table[mod][old_idx] = (prev_old + "\n" if prev_old else "") + ghost
                        # and annotate the current (new) cell
                        text = f"{base_text} [{old_dl}→{new_dl}]"
                    else:
                        # if user undid later, do nothing special
                        text = base_text
                # If only previewing, show yellow dashed state
                elif final_dl != dl:
                    text = f"{base_text} → {final_dl}"
                    prev_old = table[mod][dl - 1]
                    ghost = f"[preview_from] {base_text} → {final_dl}"
                    table[mod][dl - 1] = (prev_old + "\n" if prev_old else "") + ghost
                else:
                    text = base_text

                # put label in the (possibly overridden) deadline cell
                col_idx = final_dl - 1
                prev = table[mod][col_idx]
                table[mod][col_idx] = (prev + "\n" if prev else "") + text

    # Add exam percentage to week 15 (if module has exam share)
    if "week 15" in weeks_labels:
        idx15 = weeks_labels.index("week 15")
        for mod, (_credits, assign_pct, _contact) in st.session_state.meta.items():
            exam_pct = max(0.0, (1.0 - float(assign_pct)) * 100.0)
            if exam_pct > 0:
                prev = table[mod][idx15]
                table[mod][idx15] = (prev + "\n" if prev else "") + f"Exam ({exam_pct:.0f}%)"

    df = pd.DataFrame.from_dict(table, orient="index", columns=weeks_labels)
    df.index.name = "Module"
    return df

def _weights_for_span(d: int, style: str):
    """
    Create a distribution of weights across (deadline - release) inclusive.
    This models how work is spread across the span based on study style.
    """
    if d < 0:
        return []
    if style == "Early Starter":
        # flat split across span
        return [1/(d+1)] * (d+1)
    if style == "Steady":
        # ramp-up weights (2*(i+1) / (d^2+3d+2))
        denom = (d**2 + 3*d + 2) or 1
        return [2*(i+1)/denom for i in range(d+1)]
    if style == "Just in Time":
        # all weight at deadline
        return [0]*d + [1]
    # fallback
    return [1/(d+1)] * (d+1)

def recompute_all_weekly(study_style: str,
                         meta: dict,
                         ca_map: dict,
                         deadline_overrides: dict | None = None) -> dict:
    """
    Build 15-week hour allocations for every module, applying optional overrides:
      * meta:   module -> (credits, assignment_pct, contact_hours_per_week)
      * ca_map: module -> list of (idx, weight%, deadline, release)
      * deadline_overrides: {(module, ca_idx) -> new_deadline}
    Returns dict: module -> [week1..week15 hours]
    """
    deadline_overrides = deadline_overrides or {}
    weeks15_by_mod = {}

    for mod in meta:
        credits, assign_pct, contact = meta[mod]
        total_notional = credits * 10

        # Baseline teaching hours: 1–6 contact, 7 = 0 (reading), 8–12 contact, 13–15 initially 0
        weekly = [contact]*6 + [0] + [contact]*5 + [0, 0, 0]

        # Coursework effort still to allocate within 1..12 (excluding week 7)
        prep_time = max(total_notional * assign_pct - sum(weekly[:12]), 0.0)

        ca_list = ca_map.get(mod, [])
        total_pct = sum(w for (_idx, w, _dl, _rl) in ca_list) or 1.0

        for (idx, wt, dl, rl) in ca_list:
            # apply temporary (preview) or applied override to deadline if any
            dl = deadline_overrides.get((mod, idx), dl)

            d = dl - rl
            if d < 0 or rl < 1 or dl > 12:
                # skip invalid spans or out-of-teaching-range deadlines
                continue

            T = prep_time * (wt / total_pct)
            weights = _weights_for_span(d, study_style)

            # enforce special weeks: no teaching in 7, exam weeks separated
            weekly[6] = 0
            weekly[-3:] = [0, 0, 0]

            # add distributed CA work into weeks 1..12 (excluding week 7)
            for i, w in enumerate(weights):
                week_idx = rl - 1 + i
                if 0 <= week_idx < 12 and week_idx != 6:
                    weekly[week_idx] += T * w

        # Spread exam effort across weeks 13..15 (based on exam share)
        exam_pct = 1.0 - assign_pct
        exam_effort = total_notional * exam_pct
        d_exam = 2  # (15 - 13)
        exam_w = _weights_for_span(d_exam, study_style)
        for i, w in enumerate(exam_w):
            weekly[12 + i] += exam_effort * w

        weeks15_by_mod[mod] = weekly

    return weeks15_by_mod

def total_cv_percent(weeks15_by_mod: dict) -> float:
    """
    Compute “CV%” across TOTAL hours for weeks 1..12 (ignoring week 7).
    This is the app’s “Pain score” metric (higher = more uneven workload).
    """
    totals = [0.0]*12
    for weekly in weeks15_by_mod.values():
        for i in range(12):
            totals[i] += weekly[i]
    mean = sum(totals)/12 if totals else 0.0
    if mean == 0:
        return 0.0
    # population variance/std, then CV% = std/mean * 100
    var = sum((x - mean)**2 for x in totals) / 12
    std = var ** 0.5
    return (std / mean) * 100.0

def persist_cas(name, credits, assign_pct, ca_nms, ca_wts, ca_dls, ca_rels):
    """
    Save/replace a module’s CA definitions:
      * Validates release <= deadline <= 12.
      * Updates in-memory session structures.
      * Persists CA list and schedule to DB.
    """
    notional  = credits * assign_pct * 10
    weekly    = st.session_state.baseline[name].copy()
    prep_time = max(notional - sum(weekly[:12]), 0.0)
    total_pct = sum(ca_wts) or 1.0

    cas_list = []
    for idx, (wt, dl, rel, nm) in enumerate(zip(ca_wts, ca_dls, ca_rels, ca_nms), start=1):
        if (dl - rel) < 0 or rel < 1 or dl > 12:
            st.warning(f"Invalid release/deadline for CA#{idx} — skipped.")
            continue
        cas_list.append((idx, wt, dl, rel))
        weekly[dl - 1] += prep_time * (wt / total_pct)
        st.session_state.ca_names.setdefault(name, {})[idx] = nm

    # Enforce week 7 = 0 and exam weeks separate
    weekly[6]   = 0
    weekly[-3:] = [0, 0, 0]

    # Update session & persist to DB
    st.session_state.modules[name] = weekly
    st.session_state.ca_map[name]  = cas_list
    save_cas(name, cas_list)
    save_schedule(name, weekly)

# ───────────────────────────────────────────────────────────────────────────────
# 5) CACHE LAYER FOR SCENARIOS
# Memoize recomputation for speed; scenario generator tries ±1 week shifts.
# ───────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _weeks15_cached(study_style, meta_items, ca_map_items, overrides_items):
    """Cached call to recompute_all_weekly with hashable inputs."""
    meta = dict(meta_items)
    ca_map = {k: list(v) for k, v in ca_map_items}
    overrides = dict(overrides_items) if overrides_items else None
    return recompute_all_weekly(study_style, meta, ca_map, overrides)

def _hashables_for_cache(meta, ca_map, overrides=None):
    """
    Convert dicts to sorted tuples so cache keys are stable & hashable.
    """
    meta_items = tuple(sorted((k, tuple(v)) for k, v in meta.items()))
    ca_map_items = tuple(sorted((k, tuple(tuple(x) for x in v)) for k, v in ca_map.items()))
    overrides_items = tuple(sorted(overrides.items())) if overrides else None
    return meta_items, ca_map_items, overrides_items

def generate_scenarios_exact_upto_k(all_cas, Kmax, study_style, meta, ca_map, valid_fn):
    """
    Build *candidate* scenarios where exactly k CAs are shifted (k=1..Kmax).
    For each chosen CA, try shifting ±1 week (if valid per valid_fn).
    Returns list of tuples: (no_shifts, CV, changes_str)
    """
    scenarios = []
    if not all_cas or Kmax <= 0:
        return scenarios

    meta_items, ca_map_items, _ = _hashables_for_cache(meta, ca_map, None)

    for k in range(1, Kmax + 1):
        for idxs in itertools.combinations(range(len(all_cas)), k):
            for dirs in itertools.product([-1, 1], repeat=k):
                overrides, changes, valid = {}, [], True
                for pos, dir_ in zip(idxs, dirs):
                    mod, idx, rl, dl = all_cas[pos]
                    new_dl = dl + dir_
                    if not valid_fn(rl, new_dl):
                        valid = False
                        break
                    overrides[(mod, idx)] = new_dl
                    changes.append(f"{mod} CA#{idx}@week {dl}→{new_dl}")
                if not valid:
                    continue
                _, _, overrides_items = _hashables_for_cache(meta, ca_map, overrides)
                weeks15 = _weeks15_cached(study_style, meta_items, ca_map_items, overrides_items)
                scenarios.append((k, total_cv_percent(weeks15), "none" if not changes else "; ".join(changes)))
    return scenarios

# ───────────────────────────────────────────────────────────────────────────────
# Redirect any legacy “step 2” to the Results page (step 3)
# ───────────────────────────────────────────────────────────────────────────────
if st.session_state.step == 2:
    st.session_state.step = 3
    st.rerun()

# ───────────────────────────────────────────────────────────────────────────────
# 6) STEP 0 — MASTER LIST OF MODULES
# Show existing modules (from DB), allow adding a new one or editing.
# ───────────────────────────────────────────────────────────────────────────────
if st.session_state.step == 0:
    st.title("All Modules")
    c1, c2 = st.columns([3,1])
    with c2:
        # Start a new module setup
        if st.button("➕ Add New Module"):
            st.session_state.selected = "__new__"
            st.session_state["_name"] = ""
            st.session_state["_credits"] = 0.0
            st.session_state["_assign_pct"] = 0.0
            st.session_state["_contact"] = 0.0
            st.session_state["_n_ca"] = 0
            # Clear any leftover CA inputs from previous sessions
            for k in list(st.session_state.keys()):
                if k.startswith(("wt","dl","rel","nm")):
                    del st.session_state[k]
            st.session_state.step = 1
            st.rerun()

    # List modules with summary and “Edit” action
    for mod in st.session_state.meta:
        with st.expander(mod):
            cr, ap, ct = st.session_state.meta[mod]  # credits, assignment %, contact hrs/wk
            exam_pct = 1.0 - ap
            st.markdown(
                f"**Credits:** {cr}  •  "
                f"**CW %:** {ap*100:.0f}%  •  "
                f"**Exam %:** {exam_pct*100:.0f}%  •  "
                f"**Contact:** {ct}h/wk"
            )
            # Edit kicks to Step 1 with prefilled fields
            if st.button("✏️ Edit", key=f"edit_{mod}"):
                for k in list(st.session_state.keys()):
                    if k.startswith(("wt", "dl", "rel", "nm")) or k in {"_name","_credits","_assign_pct","_contact","_n_ca"}:
                        del st.session_state[k]
                st.session_state.selected = mod
                st.session_state.step = 1
                st.rerun()

    st.write("Click ‘Add New Module’ or ‘Edit’ to begin.")

# ───────────────────────────────────────────────────────────────────────────────
# 7) STEP 1 — MODULE DEFINITIONS + CAs
# Create/edit a module; fill in meta and continuous assessments.
# ───────────────────────────────────────────────────────────────────────────────
elif st.session_state.step == 1:
    st.title("Step 1 of 2: Module Definitions")

    # Pre-fill the edit form if editing existing module (only populate once)
    if st.session_state.selected and st.session_state.selected != "__new__":
        sel = st.session_state.selected
        cr, ap, ct = st.session_state.meta[sel]

        # top-level module fields (stored in session under underscored names)
        if "_name" not in st.session_state:        st.session_state["_name"] = sel
        if "_credits" not in st.session_state:     st.session_state["_credits"] = cr
        if "_assign_pct" not in st.session_state:  st.session_state["_assign_pct"] = ap * 100
        if "_contact" not in st.session_state:     st.session_state["_contact"] = ct
        if "_n_ca" not in st.session_state:        st.session_state["_n_ca"] = len(st.session_state.ca_map.get(sel, []))

        # CA fields (persisted as wtN, dlN, relN, nmN)
        for idx, wt, dl, rel in st.session_state.ca_map.get(sel, []):
            i = idx - 1
            if f"wt{i}"  not in st.session_state: st.session_state[f"wt{i}"]  = wt
            if f"dl{i}"  not in st.session_state: st.session_state[f"dl{i}"]  = dl
            if f"rel{i}" not in st.session_state: st.session_state[f"rel{i}"] = rel
            if f"nm{i}"  not in st.session_state:
                st.session_state[f"nm{i}"] = st.session_state.ca_names.get(sel, {}).get(idx, "")

    # Main module form
    with st.form("module_form"):
        name       = st.text_input("Module name", value=st.session_state.get("_name",""))
        credits    = st.number_input("Credits", min_value=0.0, step=0.5,
                                     value=st.session_state.get("_credits",0.0))
        # Legacy field kept; you also expose CW% / Exam% below and then overwrite this
        assign_pct = st.number_input("Assignment % of total hours",
                                     min_value=0.0, max_value=100.0,
                                     value=st.session_state.get("_assign_pct",0.0))/100.0
        
        # Coursework vs Exam — user-facing sliders; assigns cw_pct back to assign_pct
        colA, colB = st.columns(2)
        with colA:
            cw_pct = st.number_input(
                "Coursework % of total hours",
                min_value=0, max_value=100, step=1,
                value=int(assign_pct * 100)
            ) / 100.0
        with colB:
            st.number_input(
                "Exam % of total hours",
                min_value=0, max_value=100, step=1,
                value=int((1 - cw_pct) * 100),
                disabled=True
            )
        assign_pct = cw_pct  # overwrite with the intended value from the paired inputs

        contact    = st.number_input("Contact hrs/week",
                                     min_value=0.0, step=0.5,
                                     value=st.session_state.get("_contact",0.0))

        n_ca = st.number_input("How many CAs?", min_value=0, step=1,
                               value=st.session_state.get("_n_ca",0))
        
        # Collect CA inputs
        ca_nms, ca_wts, ca_dls, ca_rels = [], [], [], []

        if n_ca > 0:
            st.subheader("Continuous Assessments")
            for i in range(int(n_ca)):
                # CA name (optional)
                nm = st.text_input(
                    f"Name for CA #{i+1}",
                    key=f"nm{i}",
                    value=st.session_state.get(f"nm{i}", "")
                )
                c1, c2, c3 = st.columns([2, 1, 1])
                with c1:
                    w = st.number_input(
                        f"Weight % for CA #{i+1}",
                        min_value=0.0, max_value=100.0,
                        key=f"wt{i}",
                        value=st.session_state.get(f"wt{i}", 0.0)
                    )
                with c2:
                    d = st.number_input(
                        f"Deadline week for CA #{i+1}",
                        min_value=1, max_value=12, step=1,
                        key=f"dl{i}",
                        value=st.session_state.get(f"dl{i}", 1)
                    )
                with c3:
                    r = st.number_input(
                        f"Release week for CA #{i+1}",
                        min_value=1, max_value=12, step=1,
                        key=f"rel{i}",
                        value=st.session_state.get(f"rel{i}", 1)
                    )
                ca_nms.append(nm); ca_wts.append(w); ca_dls.append(d); ca_rels.append(r)

        # Form submit buttons
        b1, b2, b3 = st.columns(3)
        with b1: back = st.form_submit_button("◀ Back to list")
        with b2: save_mod = st.form_submit_button("Save / Update Module")
        with b3: alloc_cas = st.form_submit_button("Allocate CAs")

    # Normalize booleans if not created (streamlit re-run guard)
    back = bool(back) if 'back' in locals() else False
    save_mod = bool(save_mod) if 'save_mod' in locals() else False
    alloc_cas = bool(alloc_cas) if 'alloc_cas' in locals() else False

    if back:
        # Return to module list without saving
        go_home()
        st.stop()

    if save_mod:
        # Save the module meta (and baseline schedule) to DB
        if not name:
            st.error("Module name is required.")
        else:
            # Build baseline schedule: contact hours; week 7 = 0; 13–15 = 0 here
            sched = [contact]*6 + [0] + [contact]*5 + [0, 0, 0]

            # Update session
            st.session_state.baseline[name] = sched.copy()
            st.session_state.modules[name]  = sched.copy()
            st.session_state.meta[name]     = (credits, assign_pct, contact)

            # Persist meta + schedule
            save_module_meta(name, credits, assign_pct, contact)
            save_schedule(name, sched)

            # If CA rows were entered, persist them too
            if int(n_ca) > 0:
                persist_cas(name, credits, assign_pct, ca_nms, ca_wts, ca_dls, ca_rels)

            st.success(f"Module '{name}' saved.")
            st.session_state.selected = None
            st.session_state.step = 0   # go back to module list
            st.rerun()

    if alloc_cas:
        # Allow allocating/overwriting CAs for an already-saved module
        if name not in st.session_state.modules:
            st.error("Please Save module first.")
        elif int(n_ca) == 0:
            st.warning("Set How many CAs? > 0 first.")
        else:
            persist_cas(name, credits, assign_pct, ca_nms, ca_wts, ca_dls, ca_rels)
            st.success(f"CAs allocated for '{name}'.")
            st.session_state.selected = None
            st.session_state.step = 0
            st.rerun()

# ───────────────────────────────────────────────────────────────────────────────
# 8) STEP 3 — RESULTS
# Compute workloads, show table/chart + recommendations + calendar.
# ───────────────────────────────────────────────────────────────────────────────
elif st.session_state.step == 3:

    st.title("Step 2 of 2: Results")
    weeks = st.session_state.weeks
    teaching_weeks = weeks[:12]  # restrict to 1..12 for the Pain score metric

    # Study Style selector controls how CA work is distributed across release..deadline
    study_style = st.selectbox(
        "Study Style",
        ["Early Starter", "Steady", "Just in Time"],
        index=["Early Starter", "Steady", "Just in Time"].index(
            st.session_state.get("study_style", "Just in Time")
        )
    )
    st.session_state["study_style"] = study_style
    st.caption("This controls how effort is distributed from CA release to deadline during allocation.")

    # Recompute module-by-module weekly workload based on the study style
    df_rows = {}
    for mod in st.session_state.meta:
        credits, assign_pct, contact = st.session_state.meta[mod]
        baseline = [contact]*6 + [0] + [contact]*5 + [0, 0, 0]
        weekly = baseline.copy()
        total_notional = credits * 10
        prep_time = max(total_notional * assign_pct - sum(weekly[:12]), 0.0)

        ca_list = st.session_state.ca_map.get(mod, [])
        total_pct = sum(w for (_, w, _, _) in ca_list) or 1.0

        for idx, wt, dl, rel in ca_list:
            T = prep_time * (wt / total_pct)
            d = dl - rel
            if d < 0 or rel < 1 or dl > 12:
                continue

            # Distribute work across release..deadline per chosen style
            if study_style == "Early Starter":
                weights = [1 / (d + 1)] * (d + 1)
            elif study_style == "Steady":
                denom = (d**2 + 3*d + 2) or 1
                weights = [2 * (i + 1) / denom for i in range(d + 1)]
            else:
                weights = [0] * d + [1]

            weekly[6] = 0            # enforce week 7 = 0
            weekly[-3:] = [0, 0, 0]  # keep 13–15 separate

            for i, w in enumerate(weights):
                week_idx = rel - 1 + i
                if 0 <= week_idx < 12 and week_idx != 6:  # exclude week 7
                    weekly[week_idx] += T * w

        # Spread exam effort across weeks 13..15
        exam_pct = 1.0 - assign_pct
        exam_effort = total_notional * exam_pct
        d_exam = 2
        if study_style == "Early Starter":
            weights = [1 / (d_exam + 1)] * (d_exam + 1)
        elif study_style == "Steady":
            denom = (d_exam**2 + 3*d_exam + 2) or 1
            weights = [2 * (i + 1) / denom for i in range(d_exam + 1)]
        else:
            weights = [0] * d_exam + [1]
        for i, w in enumerate(weights):
            weekly[12 + i] += exam_effort * w

        df_rows[mod] = weekly

    # Build DataFrame for display: rows=modules (+ TOTAL), cols=weeks 1..15
    df_main = pd.DataFrame(df_rows, index=weeks).T
    tot      = df_main[weeks].sum(axis=0)
    df_total = pd.DataFrame([tot.values], index=["TOTAL"], columns=weeks)
    df       = pd.concat([df_main, df_total], axis=0)

    # Add totals (1..12) and the “Pain score” metric = CV% over weeks 1..12
    df["Total"] = df[teaching_weeks].sum(axis=1)
    df["Pain Score"]    = (df[teaching_weeks].std(ddof=0, axis=1) / df[teaching_weeks].mean(axis=1) * 100).round(1)
    df["Colour"]= df["Pain Score"].map(colour_for_cv)

    # NOTE: Below, the styled table references "Pain score" (lowercase s) in one place.
    # That will not match the column just created ("Pain Score") and may error.
    # Leaving code unchanged per your request—just flagging it for you.

    # Toggle to show/hide the workload table
    if "show_workload" not in st.session_state:
        st.session_state.show_workload = False

    col_show, _ = st.columns([1, 7])
    if col_show.button(("Hide" if st.session_state.show_workload else "Show") + " workload table"):
        st.session_state.show_workload = not st.session_state.show_workload

    if st.session_state.show_workload:
        df_disp = df.round(1)
        styled = (
            df_disp[weeks + ["Total","Pain Score","Colour"]]
            .style
            .applymap(lambda v: f"background-color: {colour_for_cv(v)}", subset=["Pain Score"])
            .set_properties(color="red", subset=pd.IndexSlice[:, ["week 13","week 14","week 15"]])
        )
        st.markdown("<div style='overflow-x:auto'>" + styled.to_html() + "</div>", unsafe_allow_html=True)
    else:
        # keep df_disp defined for Excel download
        df_disp = df.round(1)

    # ---- Scenario computation (for Excel sheet) ----
    def _valid_shift(rl: int, new_dl: int) -> bool:
        """Only allow deadlines within 1..12 and not earlier than release."""
        return 1 <= new_dl <= 12 and new_dl >= rl

    _all_cas = []
    for _m, _cas_list in st.session_state.ca_map.items():
        for (_i, _w, _dl, _rl) in _cas_list:
            if 1 <= _dl <= 12:
                _all_cas.append((_m, _i, _rl, _dl))

    # Keep Excel enumeration modest (k up to 3)
    _excel_Kmax = min(3, len(_all_cas))
    _scenarios = generate_scenarios_exact_upto_k(
        _all_cas, _excel_Kmax, study_style,
        st.session_state.meta, st.session_state.ca_map, _valid_shift
    )

    # Data for Excel “All Scenarios” sheet
    out = (pd.DataFrame(_scenarios, columns=["no_shifts","CV","changes"])  # keeps old column names for file
           if _scenarios else pd.DataFrame(columns=["no_shifts","CV","changes"]))
    out = out.sort_values(["CV","no_shifts"], ascending=[True,True]).reset_index(drop=True)

    # ─── Heatmaps entry point (jump to ALL pairs view) ───────────────────
    if st.button("Show all module-pair heatmaps"):
        st.session_state.heatmap_modules = "__ALL__"
        st.session_state.step = "HEATMAP"
        st.rerun()

    # ─── Calendar summary + current “Pain score” ─────────────────────────
    st.markdown("### Calendar")
    base_weeks15 = recompute_all_weekly(study_style, st.session_state.meta, st.session_state.ca_map)
    base_cv = total_cv_percent(base_weeks15)
    st.markdown(f"<p class='caption-lg'>Pain score (weeks 1–12): <b>{base_cv:.1f}%</b></p>", unsafe_allow_html=True)

    # Build calendar DF and style it for visual highlighting
    cal_df = build_calendar_df(st.session_state.weeks)
    cal_df_display = cal_df.reset_index()
    week_cols = [c for c in cal_df_display.columns if c.startswith("week")]

    n_weeks = len(week_cols)
    module_pct = 16                 # width % for the Module column
    week_pct   = (100 - module_pct) / n_weeks

    # CSS to make the calendar large, scrollable, and readable
    st.markdown("""
    <style>
    .block-container {padding-left: 1rem; padding-right: 1rem;}
    .calendar-wrap {height: calc(100vh - 220px); overflow:auto;}
    .calendar-wrap table {table-layout: fixed; width: 100%; border-collapse: collapse;}
    .calendar-wrap th, .calendar-wrap td {box-sizing: border-box;}
    .calendar-wrap th {font-size: 15px; padding: 10px 8px;}
    .calendar-wrap td {font-size: 14px; padding: 12px 10px; line-height: 1.3; vertical-align: top;}
    @media (max-width: 1400px){
      .calendar-wrap th {font-size: 14px; padding: 8px 6px;}
      .calendar-wrap td {font-size: 13px; padding: 10px 6px;}
    }
    </style>
    """, unsafe_allow_html=True)

    # If user is previewing overrides (from Recommendations), use them
    preview = st.session_state.get("preview_overrides", {})
    weeks15_by_mod = recompute_all_weekly(study_style, st.session_state.meta, st.session_state.ca_map,
                                          preview if preview else None)

    # Helper for per-module “Pain score” colour stripe on the left border
    def _cv_percent_1to12(weekly):
        vals = weekly[:12]
        m = sum(vals)/12 if vals else 0.0
        if m == 0: return 0.0
        var = sum((x - m)**2 for x in vals) / 12
        return (var ** 0.5) / m * 100.0

    band_hex = {"green":"#2ecc71","yellow":"#f1c40f","orange":"#e67e22","red":"#e74c3c","black":"#000000"}
    row_band = {m: colour_for_cv(_cv_percent_1to12(w)) for m,w in weeks15_by_mod.items()}

    # Style each cell based on content and band, add ghost/preview visuals
    def _color_calendar(col: pd.Series) -> list[str]:
        styles = []
        col_name = col.name
        for i, v in enumerate(col):
            txt = v if isinstance(v, str) else ""
            module = cal_df_display.iloc[i]["Module"]
            stripe = band_hex.get(row_band.get(module, "green"), "#2ecc71")
            style = f"border-left: 6px solid {stripe};"
            if col_name == "week 7":
                style += " background-color:#3a3a3a; color:#eaeaea;"
            elif col_name == "week 15" and "Exam" in txt:
                style += " background-color:#ede7f6; color:#111;"
            else:
                if txt.strip():
                    if "[moved_from]" in txt:
                        style += (" background: repeating-linear-gradient(45deg,#3f3f3f, #3f3f3f 8px, #333 8px, #333 16px);"
                                  " color:#eaeaea; border:2px solid #ff7043; border-radius:4px;")
                    elif "[preview_from]" in txt:
                        style += " background-color:#fff3cd; color:#111; outline:2px dashed #856404; outline-offset:-2px;"
                    elif "[" in txt and "→" in txt and "]" in txt:
                        style += " background-color:#bbdefb; color:#111; box-shadow: inset 0 0 0 2px #1e88e5;"
                    elif "\n" in txt:
                        style += " background-color:#ffcc80; color:#111;"
                    else:
                        style += " background-color:#e3f2fd; color:#111;"
            styles.append(style)
        return styles

    # Left border colour bar per module (based on its band)
    def _stripe_module_col(col: pd.Series) -> list[str]:
        styles = []
        for i, _ in enumerate(col):
            module = cal_df_display.iloc[i]["Module"]
            stripe = band_hex.get(row_band.get(module, "green"), "#2ecc71")
            styles.append(f"border-left: 8px solid {stripe};")
        return styles

    # Render the calendar table as styled HTML (for more control than st.dataframe)
    calendar_styled = (
        cal_df_display.style
        .hide(axis="index")
        .set_table_styles([
            {"selector":"table","props":[("table-layout","fixed"),("width","100%"),("border-collapse","collapse")]},
            {"selector":"th","props":[("font-size","15px"),("padding","10px 8px")]},
            {"selector":"td","props":[("font-size","14px"),("padding","12px 10px"),("line-height","1.3"),("vertical-align","top")]}
        ], overwrite=True)
        .set_properties(subset=pd.IndexSlice[:, ["Module"]], **{"width": f"{module_pct}%"} )
        .set_properties(subset=pd.IndexSlice[:, week_cols],   **{"width": f"{week_pct:.4f}%"} )
        .apply(_color_calendar, axis=0, subset=week_cols)
        .apply(_stripe_module_col, axis=0, subset=["Module"])
    )
    st.markdown(f"<div class='calendar-wrap'>{calendar_styled.to_html()}</div>", unsafe_allow_html=True)
    st.caption("Legend: left border colour = CV band (green/yellow/orange/red/black); pale blue = CA; deeper amber = multiple CAs; yellow (dashed) = preview move; purple = exam; grey = week 7.")

    # ── Recommendations UI (choose number of shifts to consider) ──
    st.subheader("Recommendations")

    if "N" not in st.session_state:
        st.session_state.N = 2          # default number of shifts to evaluate

    col_num, col_btn, _ = st.columns([1.2, 0.8, 6])
    with col_num:
        N_typed = st.number_input(
            "Type N (1–10)",
            min_value=1, max_value=10, step=1,
            value=int(st.session_state.N),
            key="N_input"
        )
    with col_btn:
        if st.button("Show", key="showN"):
            st.session_state.N = int(N_typed)
            st.rerun()

    # Undo box appears after you apply a set of shifts
    if st.session_state.get("undo_payload"):
        with st.container(border=True):
            st.write("**Undo last shift**")
            if st.session_state.get("undo_caption"):
                st.caption(st.session_state.undo_caption)
            if st.button("Undo"):
                # Revert deadlines for each CA in the undo payload
                for (m, idx), old_dl in st.session_state.undo_payload.items():
                    old_list = st.session_state.ca_map.get(m, [])
                    new_list = []
                    for (j, wt, dl, rl) in old_list:
                        if j == idx:
                            dl = old_dl
                        new_list.append((j, wt, dl, rl))
                    st.session_state.ca_map[m] = new_list
                    save_cas(m, new_list)
                st.session_state.last_applied_moves = {}
                st.session_state.undo_payload = None
                st.session_state.undo_caption = ""
                st.success("Reverted last shift.")
                st.rerun()

    # Build the set of adjustable CAs (1..12 only)
    all_cas = []
    for mod, cas_list in st.session_state.ca_map.items():
        for (idx, wt, dl, rl) in cas_list:
            if 1 <= dl <= 12:
                all_cas.append((mod, idx, rl, dl))

    # Generate scenarios for k up to N, then we’ll filter to exactly N
    scenarios = generate_scenarios_exact_upto_k(
        all_cas, st.session_state.N, study_style,
        st.session_state.meta, st.session_state.ca_map, _valid_shift
    )

    st.subheader("Recommendations")

    if scenarios:
        # Build df and filter to rows with exactly N shifts
        df_scen = pd.DataFrame(scenarios, columns=["no_shifts","Pain score","changes"])
        filtered = df_scen.query("no_shifts == @st.session_state.N")

        # NOTE: Next line sorts by "CV" which no longer exists in df_scen.
        # Leaving intact per request, but this will error at runtime.
        out_visible = (
        filtered
        .sort_values(["Pain score","no_shifts"], ascending=[True, True])
        .reset_index(drop=True)
        )

        topN = out_visible.head(5).copy()

        # For each top option, show “Visualize” (preview) and “Shift now” (apply)
        for i, row in topN.iterrows():
            with st.container(border=True):
                st.write(f"**Option {i+1}** — Pain score **{row['Pain score']:.1f}%**")
                st.caption(row["changes"] if row["changes"] != "none" else "No changes")
                c1, c2, _ = st.columns([1,1,6])

                # Unique key to avoid Streamlit button ID collisions
                uniq = f"{i}_{abs(hash(row['changes'])) % 10_000_000}"

                # Preview: store overrides in session (yellow dashed calendar)
                if c1.button("Visualize", key=f"viz_cal_{uniq}"):
                    st.session_state["preview_overrides"] = {
                        (p.split(' CA#')[0].strip(), int(p.split(' CA#')[1].split('@')[0])):
                        int(p.split('→')[1].strip())
                        for p in [x.strip() for x in row["changes"].split(';')] if p and "→" in p
                    }
                    st.rerun()

                # Apply: mutate ca_map deadlines, persist to DB, enable Undo
                if c2.button("Shift now", key=f"apply_cal_{uniq}"):
                    overrides = {
                        (p.split(' CA#')[0].strip(), int(p.split(' CA#')[1].split('@')[0])):
                        int(p.split('→')[1].strip())
                        for p in [x.strip() for x in row["changes"].split(';')] if p and "→" in p
                    }
                    if not overrides:
                        st.info("No changes to apply.")
                    else:
                        # Prepare undo and record last moves
                        undo_map, last_moves = {}, {}
                        for (m, idx), new_dl in overrides.items():
                            for (j, wt, dl, rl) in st.session_state.ca_map.get(m, []):
                                if j == idx:
                                    undo_map[(m, idx)] = dl
                                    last_moves[(m, idx)] = (dl, new_dl)
                                    break
                        st.session_state.undo_payload = undo_map
                        st.session_state.undo_caption = row["changes"]
                        st.session_state.last_applied_moves = last_moves

                        # Apply overrides to the CA map and persist
                        for (m, idx), new_dl in overrides.items():
                            old_list = st.session_state.ca_map.get(m, [])
                            new_list = []
                            for (j, wt, dl, rl) in old_list:
                                if j == idx:
                                    dl = new_dl
                                new_list.append((j, wt, dl, rl))
                            st.session_state.ca_map[m] = new_list
                            save_cas(m, new_list)

                        # Clear preview and refresh
                        st.session_state.pop("preview_overrides", None)
                        st.success("Shifts applied. Calendar shows old location shaded with an arrow old→new.")
                        st.rerun()
    else:
        st.info("No candidate scenarios available with ±1 week shifts.")

    # If preview overrides exist, allow clearing them
    if st.session_state.get("preview_overrides"):
        if st.button("Reset preview"):
            st.session_state.pop("preview_overrides", None)
            st.rerun()

    # ---- Download Excel file with two sheets: Workload + All Scenarios ----
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_disp.to_excel(writer, sheet_name="Workload")
        out.to_excel(writer, sheet_name="All Scenarios", index=False)
    st.download_button(
        " Download Excel",
        data=buf.getvalue(),
        file_name="results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # ---- Optional: Workload line chart (TOTAL highlighted thicker) ----
    chart_df = (
        df[weeks]
        .reset_index()
        .melt(id_vars="index", var_name="Week", value_name="Hours")
        .rename(columns={"index":"Module"})
    )
    st.subheader("Workload Over Time")
    chart = (
        alt.Chart(chart_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("Week:N", sort=weeks, axis=alt.Axis(labelAngle=-45)),
            y="Hours:Q",
            color=alt.Color("Module:N", legend=alt.Legend(title="Module")),
            size=alt.condition(alt.datum.Module=="TOTAL", alt.value(4), alt.value(2))
        )
        .properties(width=700, height=400)
    )
    st.altair_chart(chart, use_container_width=True)

# ───────────────────────────────────────────────────────────────────────────────
# 9) STEP "HEATMAP" — ALL-PAIRS VIEW (and fallback single-pair logic stub)
# Draw 12x12 cross heatmaps for every pair of modules with CAs in teaching weeks.
# ───────────────────────────────────────────────────────────────────────────────
elif st.session_state.step == "HEATMAP":
    flag = st.session_state.get("heatmap_modules")
    style = st.session_state.get("study_style", "Just in Time")

    # ------- ALL-PAIRS PAGE -------
    if flag == "__ALL__":

        # Filter modules that have at least one CA deadline in weeks 1..12
        def _has_teaching_ca(mod):
            return any(1 <= dl <= 12 for (_i, _w, dl, _r) in st.session_state.ca_map.get(mod, []))

        mods_with_cas = [m for m in st.session_state.meta.keys() if _has_teaching_ca(m)]
        pairs = list(itertools.combinations(mods_with_cas, 2))

        st.title("All module-pair heatmaps")

        # Inner function to compute and/or render one pair’s lattice
        def render_pair(A_mod, B_mod, highlight_coords=None, compute_only=False):
            # Show heading unless we’re in the compute-only pass (for global min)
            if not compute_only:
                st.markdown(f"### {A_mod} vs {B_mod} — CV cross heatmap")

            # Extract CAs for both modules within weeks 1..12
            A_cas = [(idx, wt, dl, rl) for (idx, wt, dl, rl) in st.session_state.ca_map.get(A_mod, []) if 1 <= dl <= 12]
            B_cas = [(idx, wt, dl, rl) for (idx, wt, dl, rl) in st.session_state.ca_map.get(B_mod, []) if 1 <= dl <= 12]

            # Shading sets: which rows/cols contain any CA (for faint shading)
            A_weeks_with_ca = {dl for (_idx, _wt, dl, _rl) in A_cas}
            B_weeks_with_ca = {dl for (_idx, _wt, dl, _rl) in B_cas}

            if not A_cas or not B_cas:
                if compute_only:
                    # Skip rendering & return empty signals to caller
                    return pd.DataFrame(), None
                st.info("One or both selected modules have no CAs with deadlines in weeks 1–12.")
                return

            # Helpers for cell colouring by band and legality checks
            def cv_band_and_color(cv: float):
                if cv < 40:           return "<40",      "#2ecc71"
                elif cv < 50:         return "40–49.9",  "#f1c40f"
                elif cv < 65:         return "50–64.9",  "#e67e22"
                elif cv < 80:         return "65–79.9",  "#e74c3c"
                else:                 return "≥80",      "#b71c1c"

            def can_shift(mod: str, w: int, dir_: int) -> bool:
                # legal if new week within 1..12 and not before release for any CA at that week
                new_w = w + dir_
                if not (1 <= new_w <= 12):
                    return False
                for (_idx, _wt, dl, rl) in st.session_state.ca_map.get(mod, []):
                    if dl == w and new_w < rl:
                        return False
                return True

            # Find actual collision weeks (same-week deadlines between A and B)
            A_dead = {dl for (_i, _w, dl, _r) in A_cas} if A_cas else set()
            B_dead = {dl for (_i, _w, dl, _r) in B_cas} if B_cas else set()
            collisions = sorted(A_dead & B_dead)

            # Center CV (no shifts)
            base_weeks = recompute_all_weekly(style, st.session_state.meta, st.session_state.ca_map)
            center_cv = total_cv_percent(base_weeks)

            # Data store for the 12x12 lattice; we’ll keep best CV per cell
            grid = {(a, b): {"cv": None, "count": 0, "label": "", "band": ""} for a in range(1, 13) for b in range(1, 13)}

            def set_invalid(a, b):
                # mark the cell as invalid (“–”) if no label yet
                cell = grid[(a, b)]
                if cell["label"] == "":
                    cell["label"] = "–"

            def put_cv(a, b, cv_val: float):
                # store minimum CV and a text label; tie counts tracked but not shown
                band, _ = cv_band_and_color(cv_val)
                cell = grid[(a, b)]
                if cell["cv"] is None or cv_val < cell["cv"] - 1e-9:
                    cell["cv"] = cv_val
                    cell["count"] = 1
                    cell["band"] = band
                    cell["label"] = f"{cv_val:.1f}"
                elif abs(cv_val - cell["cv"]) <= 1e-9:
                    cell["count"] += 1
                    cell["label"] = f"{cell['cv']:.1f}"

            # Compute legal neighbors for each collision:
            # center (w,w), A moves ±1, B moves ±1, and both move (diagonals), plus ring-2 orthogonals
            for w in collisions:
                put_cv(w, w, center_cv)  # center

                # A moves; B stays
                for dir_ in (-1, 1):
                    a_w = w + dir_
                    if can_shift(A_mod, w, dir_):
                        overrides = {(A_mod, idx): a_w
                                    for (idx, wt, dl, rl) in st.session_state.ca_map.get(A_mod, [])
                                    if dl == w}
                        weeks_shift = recompute_all_weekly(style, st.session_state.meta, st.session_state.ca_map, overrides)
                        put_cv(a_w, w, total_cv_percent(weeks_shift))
                    else:
                        if 1 <= a_w <= 12:
                            set_invalid(a_w, w)

                # B moves; A stays
                for dir_ in (-1, 1):
                    b_w = w + dir_
                    if can_shift(B_mod, w, dir_):
                        overrides = {(B_mod, idx): b_w
                                    for (idx, wt, dl, rl) in st.session_state.ca_map.get(B_mod, [])
                                    if dl == w}
                        weeks_shift = recompute_all_weekly(style, st.session_state.meta, st.session_state.ca_map, overrides)
                        put_cv(w, b_w, total_cv_percent(weeks_shift))
                    else:
                        if 1 <= b_w <= 12:
                            set_invalid(w, b_w)

                # Both move (diagonals)
                for dA in (-1, 1):
                    for dB in (-1, 1):
                        a_w = w + dA
                        b_w = w + dB
                        if can_shift(A_mod, w, dA) and can_shift(B_mod, w, dB):
                            overA = {(A_mod, idx): a_w
                                     for (idx, wt, dl, rl) in st.session_state.ca_map.get(A_mod, [])
                                     if dl == w}
                            overB = {(B_mod, idx): b_w
                                     for (idx, wt, dl, rl) in st.session_state.ca_map.get(B_mod, [])
                                     if dl == w}
                            overrides = {}
                            overrides.update(overA)
                            overrides.update(overB)
                            weeks_shift = recompute_all_weekly(
                                style, st.session_state.meta, st.session_state.ca_map, overrides
                            )
                            put_cv(a_w, b_w, total_cv_percent(weeks_shift))
                        else:
                            if 1 <= a_w <= 12 and 1 <= b_w <= 12:
                                set_invalid(a_w, b_w)

                # Ring-2 orthogonals (±2, 0) and (0, ±2)
                for step in (-2, 2):
                    a_w = w + step
                    if can_shift(A_mod, w, step):
                        overA = {(A_mod, idx): a_w
                                 for (idx, wt, dl, rl) in st.session_state.ca_map.get(A_mod, [])
                                 if dl == w}
                        weeks_shift = recompute_all_weekly(style, st.session_state.meta, st.session_state.ca_map, overA)
                        put_cv(a_w, w, total_cv_percent(weeks_shift))
                    elif 1 <= a_w <= 12:
                        set_invalid(a_w, w)

                    b_w = w + step
                    if can_shift(B_mod, w, step):
                        overB = {(B_mod, idx): b_w
                                 for (idx, wt, dl, rl) in st.session_state.ca_map.get(B_mod, [])
                                 if dl == w}
                        weeks_shift = recompute_all_weekly(style, st.session_state.meta, st.session_state.ca_map, overB)
                        put_cv(w, b_w, total_cv_percent(weeks_shift))
                    elif 1 <= b_w <= 12:
                        set_invalid(w, b_w)

            # Build a DataFrame for the lattice cells for Altair
            all_cells = [{"A_week": a, "B_week": b} for a in range(1, 13) for b in range(1, 13)]
            rows = []
            for (a, b) in [(r["A_week"], r["B_week"]) for r in all_cells]:
                cell = grid[(a, b)]
                label = cell["label"]
                band  = cell["band"]
                if band == "<40":         color = "#2ecc71"
                elif band == "40–49.9":   color = "#f1c40f"
                elif band == "50–64.9":   color = "#e67e22"
                elif band == "65–79.9":   color = "#e74c3c"
                elif band == "≥80":       color = "#b71c1c"
                else:                     color = None
                rows.append({
                    "A_week": a, "B_week": b, "Label": label, "Band": band, "Color": color,
                    "IsDash": (label == "–"), "HasBand": band != "",
                    "RowHasA": (a in A_weeks_with_ca), "ColHasB": (b in B_weeks_with_ca),
                })

            grid_df = pd.DataFrame(rows)
            # Numeric form of the label for min calculations
            grid_df["CVnum"] = pd.to_numeric(grid_df["Label"], errors="coerce")

            # Flag the cells that represent actual collisions (diagonal & real)
            grid_df["IsCollision"] = grid_df.apply(
                lambda r: (int(r["A_week"]) == int(r["B_week"])) and (int(r["A_week"]) in collisions),
                axis=1
            )

            # Minimum CV for this pair (to compare across pairs)
            pair_min_row = grid_df.dropna(subset=["CVnum"]).nsmallest(1, "CVnum")
            pair_min_cv = float(pair_min_row.iloc[0]["CVnum"]) if not pair_min_row.empty else None

            if compute_only:
                # Return the data (no chart) so the caller can compute global minimum
                return grid_df, pair_min_cv

            # If caller passed highlight coords (global mins), mark them
            if highlight_coords:
                highlight_set = set(highlight_coords)
                grid_df["IsGlobalBest"] = grid_df.apply(
                    lambda r: (int(r["A_week"]), int(r["B_week"])) in highlight_set, axis=1
                )
            else:
                grid_df["IsGlobalBest"] = False

            band_domain = ["<40", "40–49.9", "50–64.9", "65–79.9", "≥80"]
            band_colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#b71c1c"]

            # Layout sizes for a spacious grid
            cell_size = 60
            chart_width = cell_size * 12
            chart_height = cell_size * 12
            Y_DOMAIN_DESC = list(range(12, 0, -1))  # show 12 at the top

            # Faint shading where rows/cols have at least one CA
            row_shade = (
                alt.Chart(grid_df)
                .mark_rect(opacity=0.28, fill="#42a5f5", stroke=None)
                .encode(
                    x=alt.X("A_week:O", scale=alt.Scale(domain=list(range(1,13))), axis=alt.Axis(labels=False, ticks=False)),
                    y=alt.Y("B_week:O", scale=alt.Scale(domain=Y_DOMAIN_DESC),        axis=alt.Axis(labels=False, ticks=False)),
                )
                .transform_filter("datum.RowHasA == true")
                .properties(width=chart_width, height=chart_height)
            )

            col_shade = (
                alt.Chart(grid_df)
                .mark_rect(opacity=0.24, fill="#ffb74d", stroke=None)
                .encode(
                    x=alt.X("A_week:O", scale=alt.Scale(domain=list(range(1,13))), axis=alt.Axis(labels=False, ticks=False)),
                    y=alt.Y("B_week:O", scale=alt.Scale(domain=Y_DOMAIN_DESC),        axis=alt.Axis(labels=False, ticks=False)),
                )
                .transform_filter("datum.ColHasB == true")
                .properties(width=chart_width, height=chart_height)
            )

            # Base lattice with axes
            lattice = (
                alt.Chart(grid_df)
                .mark_rect(fillOpacity=0, stroke="#9fb3c0", strokeWidth=1)
                .encode(
                    x=alt.X(
                        "A_week:O",
                        title=f"{A_mod} — deadline week (1–12)",
                        scale=alt.Scale(domain=list(range(1,13))),
                        axis=alt.Axis(orient='bottom', labelAngle=0, labelFontSize=12, titleFontSize=13, ticks=True),
                    ),
                    y=alt.Y(
                        "B_week:O",
                        title=f"{B_mod} — deadline week (1–12)",
                        scale=alt.Scale(domain=Y_DOMAIN_DESC),
                        axis=alt.Axis(orient='left', labelAngle=0, labelFontSize=12, titleFontSize=13, ticks=True),
                    ),
                )
                .properties(width=chart_width, height=chart_height)
            )

            # Banded fill by CV band
            heat = (
                alt.Chart(grid_df)
                .mark_rect(opacity=0.35, stroke=None)
                .encode(
                    x=alt.X("A_week:O",
                            scale=alt.Scale(domain=list(range(1,13))),
                            axis=alt.Axis(labels=False, ticks=False),
                    ),
                    y=alt.Y("B_week:O",
                            scale=alt.Scale(domain=Y_DOMAIN_DESC),
                            axis=alt.Axis(labels=False, ticks=False),
                    ),
                    color=alt.Color("Band:N",
                                    legend=alt.Legend(title="Pain score band (TOTAL, weeks 1–12)"),
                                    scale=alt.Scale(domain=band_domain, range=band_colors)),
                )
                .transform_filter("datum.HasBand == true")
                .properties(width=chart_width, height=chart_height)
            )

            # Numeric labels in each cell (CV values or “–”)
            text_layer = (
                alt.Chart(grid_df)
                .mark_text(fontSize=12)
                .encode(
                    x=alt.X("A_week:O",
                            scale=alt.Scale(domain=list(range(1,13))),
                            axis=alt.Axis(labels=False, ticks=False),
                    ),
                    y=alt.Y("B_week:O",
                            scale=alt.Scale(domain=Y_DOMAIN_DESC),
                            axis=alt.Axis(labels=False, ticks=False),
                    ),
                    text="Label:N",
                    color=alt.condition(
                        alt.datum.IsDash == True,
                        alt.value("#666666"),
                        alt.Color("Band:N", scale=alt.Scale(domain=band_domain, range=band_colors), legend=None),
                    ),
                )
                .transform_filter("datum.Label != ''")
                .properties(width=chart_width, height=chart_height)
            )

            # Blue rectangle around collision cells
            collision_overlay = (
                alt.Chart(grid_df)
                .mark_rect(fillOpacity=0, stroke="#1e88e5", strokeWidth=4)
                .encode(
                    x=alt.X("A_week:O",
                            scale=alt.Scale(domain=list(range(1,13))),
                            axis=alt.Axis(labels=False, ticks=False)),
                    y=alt.Y("B_week:O",
                            scale=alt.Scale(domain=Y_DOMAIN_DESC),
                            axis=alt.Axis(labels=False, ticks=False)),
                )
                .transform_filter("datum.IsCollision == true")
                .properties(width=chart_width, height=chart_height)
            )

            # Black rectangle around global-best cells (if any)
            best_overlay = (
                alt.Chart(grid_df)
                .mark_rect(fillOpacity=0, strokeOpacity=1, strokeWidth=3)
                .encode(
                    x=alt.X("A_week:O",
                            scale=alt.Scale(domain=list(range(1,13))),
                            axis=alt.Axis(labels=False, ticks=False)),
                    y=alt.Y("B_week:O",
                            scale=alt.Scale(domain=Y_DOMAIN_DESC),
                            axis=alt.Axis(labels=False, ticks=False)),
                    stroke=alt.condition(alt.datum.IsGlobalBest == True,
                                         alt.value("#000000"),
                                         alt.value(None))
                )
                .transform_filter("datum.IsGlobalBest == true")
                .properties(width=chart_width, height=chart_height)
            )

            # Duplicate axes for top/right labels
            top_axis = (
                alt.Chart(grid_df)
                .mark_rect(fillOpacity=0, strokeOpacity=0)
                .encode(
                    x=alt.X("A_week:O",
                            title=None,
                            scale=alt.Scale(domain=list(range(1,13))),
                            axis=alt.Axis(orient='top', labelAngle=0, labelFontSize=12, ticks=True),
                    ),
                    y=alt.Y("B_week:O",
                            scale=alt.Scale(domain=Y_DOMAIN_DESC),
                            axis=alt.Axis(labels=False, ticks=False),
                    ),
                )
                .properties(width=chart_width, height=chart_height)
            )

            right_axis = (
                alt.Chart(grid_df)
                .mark_rect(fillOpacity=0, strokeOpacity=0)
                .encode(
                    x=alt.X("A_week:O",
                            scale=alt.Scale(domain=list(range(1,13))),
                            axis=alt.Axis(labels=False, ticks=False),
                    ),
                    y=alt.Y("B_week:O",
                            title=None,
                            scale=alt.Scale(domain=Y_DOMAIN_DESC),
                            axis=alt.Axis(orient='right', labelAngle=0, labelFontSize=12, ticks=True),
                    ),
                )
                .properties(width=chart_width, height=chart_height)
            )

            chart = (lattice + heat + row_shade + col_shade + text_layer + top_axis + right_axis + collision_overlay + best_overlay).properties(
                width=chart_width, height=chart_height
            )

            # Tiny legend explaining the border styles
            legend_collision = (
                alt.Chart(pd.DataFrame({"label": ["Collision week"]}))
                .mark_square(size=400, filled=False, stroke="#1e88e5", strokeWidth=4)
                .encode(y=alt.Y("label:N", axis=alt.Axis(title=None)), x=alt.value(16))
            )
            legend_best = (
                alt.Chart(pd.DataFrame({"label": ["Lowest CV"]}))
                .mark_square(size=400, filled=False, stroke="#000000", strokeWidth=3)
                .encode(y=alt.Y("label:N", axis=alt.Axis(title=None)), x=alt.value(16))
            )
            legend_text = (
                alt.Chart(pd.DataFrame({"label": ["Collision week", "Lowest CV"]}))
                .mark_text(align="left", dx=28, dy=3)
                .encode(y=alt.Y("label:N"), text="label:N")
            )
            key_chart = (legend_collision + legend_best + legend_text).properties(
                width=180, height=90
            )

            full_chart = (
                alt.hconcat(chart, key_chart)
                .resolve_scale(color="independent")
                .properties(background="white")
                .configure_axis(
                    grid=True, gridColor="#e6eef2",
                    labelColor="#111111", titleColor="#111111", tickColor="#111111"
                )
                .configure_legend(
                    labelColor="#111111", titleColor="#111111"
                )
                .configure_view(
                    strokeWidth=0, fill="white"
                )
            )
            st.altair_chart(full_chart, use_container_width=False)

            if not collisions:
                st.caption("No colliding deadlines for the selected modules (weeks 1–12).")

        # If no valid pairs, tell the user; else compute and render
        if not pairs:
            st.info("No module pairs with CAs in weeks 1–12 were found.")
        else:
            # PASS 1: compute per-pair minima without rendering to find the global best
            pair_min_cv_map = {}  # (A_mod, B_mod) -> min CV for that pair (float or None)
            for A_mod, B_mod in pairs:
                grid_df, pair_min_cv = render_pair(A_mod, B_mod, compute_only=True)
                pair_min_cv_map[(A_mod, B_mod)] = pair_min_cv

            valid_mins = [v for v in pair_min_cv_map.values() if v is not None]
            global_min_cv = min(valid_mins) if valid_mins else None

            if global_min_cv is None:
                st.warning("No numeric CV values found to highlight.")
            else:
                st.success(f"🌟 Global best Pain score is **{global_min_cv:.1f}%** (highlighted in all heatmaps below).")

            # PASS 2: render all pairs; for each, highlight any cell that equals the global minimum
            tol = 1e-9
            for A_mod, B_mod in pairs:
                highlight_coords = None
                if global_min_cv is not None:
                    grid_df, _ = render_pair(A_mod, B_mod, compute_only=True)
                    hits = grid_df.loc[
                        grid_df["CVnum"].notna() & (abs(grid_df["CVnum"] - global_min_cv) <= tol),
                        ["A_week", "B_week"]
                    ]
                    if not hits.empty:
                        highlight_coords = [(int(a), int(b)) for a, b in hits.to_numpy()]

                render_pair(A_mod, B_mod, highlight_coords=highlight_coords)
                st.divider()

        if st.button("◀ Previous"):
            st.session_state.step = 3
            st.rerun()

        st.stop()

    # --- Fallback stub for single-pair heatmap (not fully implemented here) ---
    pair = st.session_state.get("heatmap_modules")
    if not pair or len(pair) != 2:
        st.error("No module pair selected.")
    else:
        A_mod, B_mod = pair

        st.markdown(f"### {A_mod} vs {B_mod} — Pain score cross heatmap")
        A_cas = [(idx, wt, dl, rl) for (idx, wt, dl, rl) in st.session_state.ca_map.get(A_mod, []) if 1 <= dl <= 12]
        B_cas = [(idx, wt, dl, rl) for (idx, wt, dl, rl) in st.session_state.ca_map.get(B_mod, []) if 1 <= dl <= 12]

        # Weeks with any CA (used for shading)
        A_weeks_with_ca = {dl for (_idx, _wt, dl, _rl) in A_cas}
        B_weeks_with_ca = {dl for (_idx, _wt, dl, _rl) in B_cas}

        # Placeholder — original Step 4 code would render here

        if not collisions:
            st.caption("No colliding deadlines for the selected modules (weeks 1–12).")

    if st.button("◀ Previous"):
        st.session_state.step = 3
        st.rerun()
