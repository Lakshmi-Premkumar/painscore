# Import Streamlit for the web UI
import streamlit as st
# Import pandas for data handling
import pandas as pd
# Import itertools for combinations and product operations
import itertools
# Import io for in-memory files (Excel buffer)
import io
# Import Altair for charts
import altair as alt
# Import Streamlit column_config (currently unused but kept)
from streamlit import column_config
# Import math (currently unused but kept)
import math
# Import os (currently unused but kept)
import os

# Bring in database helper functions defined elsewhere
from db import ensure_schema, load_all, save_module_meta, save_cas, save_schedule

# Set Streamlit page layout to use the full width
st.set_page_config(layout="wide")

# Inject small CSS to style a larger caption class
st.markdown("""
<style>
  .caption-lg{
    font-size: 1.10rem;   /* larger text */
    line-height: 1.35;    /* comfortable spacing */
    margin: .2rem 0 .7rem 0; /* small margins */
    color: inherit;       /* follow theme color */
    opacity: .95;         /* slight fade */
  }
</style>
""", unsafe_allow_html=True)

# Make sure database tables exist before we query/save anything
ensure_schema()

# Cache function that loads all data from the DB for 30 seconds
@st.cache_data(ttl=30)
def load_all_cached():
    return load_all()

# Read module metadata, CA map, and schedules from the DB (with caching)
modules_meta, cas_map_db, schedules_db = load_all_cached()

# Initialize Streamlit session state keys the app depends on (with defaults)
if 'step'     not in st.session_state: st.session_state.step     = 3   # which screen to show (default to Results)
if 'modules'  not in st.session_state: st.session_state.modules  = schedules_db.copy()  # working schedules
if 'baseline' not in st.session_state: st.session_state.baseline = schedules_db.copy()  # original schedules
if 'meta'     not in st.session_state: st.session_state.meta     = modules_meta.copy()  # module info
if 'ca_map'   not in st.session_state: st.session_state.ca_map   = cas_map_db.copy()    # CA definitions
if 'ca_names' not in st.session_state: st.session_state.ca_names = {}   # display names per CA per module
if 'weeks'    not in st.session_state:
    st.session_state.weeks = [f"week {i}" for i in range(1,16)]  # labels week 1..15
if 'selected' not in st.session_state:
    st.session_state.selected = None  # which module is being edited (if any)
if 'selected_modules' not in st.session_state:
    st.session_state.selected_modules = []  # reserved for future use

if 'selected_ca_map' not in st.session_state:
    st.session_state.selected_ca_map = {}   # reserved for future use
if 'heatmap_modules' not in st.session_state:
    st.session_state.heatmap_modules = None # selection for heatmap page
if 'last_applied_moves' not in st.session_state:
    st.session_state.last_applied_moves = {}  # remember the last applied CA shifts

# Prepare undo storage for the last batch of shifts
if 'undo_payload' not in st.session_state:
    st.session_state.undo_payload = None   # map of (module, ca_idx) -> old_deadline
if 'undo_caption' not in st.session_state:
    st.session_state.undo_caption = ""     # human-readable summary of last shifts

# Build the sidebar navigation
with st.sidebar:
    st.header("Navigate")  # section title
    nav_items = ["Results", "Modules", "Module setup", "Heatmaps (all pairs)"]  # menu items

    # Figure out which sidebar item should be preselected based on current step
    default_choice = (
        "Modules"              if st.session_state.step == 0 else
        "Module setup"         if st.session_state.step == 1 else
        "Heatmaps (all pairs)" if st.session_state.step == "HEATMAP" else
        "Results"
    )

    # Radio buttons to switch pages
    choice = st.radio(
        "Go to",
        nav_items,
        index=nav_items.index(default_choice),
        label_visibility="collapsed",
        key="nav_choice",
    )

# Translate sidebar choice into an internal step value
if choice == "Results":
    st.session_state.step = 3
elif choice == "Modules":
    st.session_state.step = 0
elif choice == "Module setup":
    st.session_state.step = 1
elif choice == "Heatmaps (all pairs)":
    st.session_state.heatmap_modules = "__ALL__"
    st.session_state.step = "HEATMAP"

# ---------- Helper functions ----------

# Return a simple color band name for a CV% value
def colour_for_cv(cv: float) -> str:
    """
    Map a CV-like metric to a band name.
    """
    if cv < 92:         return "green"
    elif cv < 102:      return "yellow"
    elif cv < 112:      return "orange"
    elif cv < 120:      return "red"
    else:               return "black"

# Return an HTML-styled badge for a CV value with color coding
def cv_badge(cv: float, label: str) -> str:
    """Create a colored badge that shows a label and a CV%."""
    band = colour_for_cv(cv)  # map cv to band
    hexmap = {"green":"#2ecc71","yellow":"#f1c40f","orange":"#e67e22","red":"#e74c3c","black":"#000000"}
    bg = hexmap.get(band, "#2ecc71")  # background color
    fg = "#111111" if band != "black" else "#ffffff"  # text color
    return (
        f"<span style='display:inline-block;padding:6px 10px;"
        f"border-radius:8px;background:{bg};color:{fg};font-weight:600;"
        f"margin-right:8px;border:1px solid rgba(0,0,0,.08)'>"
        f"{label}: {cv:.1f}%</span>"
    )

# (Unused helper) Convert CV to a 0..1 range variant
def normalize_cv(ser):
    """
    Normalize coefficient of variation: CV / (1 + CV).
    """
    raw = ser.std(ddof=0) / ser.mean()
    return raw / (1 + raw)

# Move to next step
def go_next(): st.session_state.step += 1
# Move to previous step
def go_prev(): st.session_state.step -= 1
# Go back to module list and clear selection
def go_home():
    st.session_state.step = 0
    st.session_state.selected = None

# Build a table of CA rows for a module (1..12 deadlines only)
def ca_df(mod):
    """
    Return a dataframe with CA details for a given module.
    """
    cas = st.session_state.ca_map.get(mod, [])
    name_map = st.session_state.ca_names.get(mod, {})
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

# Return the set of weeks (1..12) where the module has CA deadlines
def deadline_set(mod):
    cas = st.session_state.ca_map.get(mod, [])
    return {dl for (_idx, _wt, dl, _rl) in cas if 1 <= dl <= 12}

# Build the calendar-like dataframe used for the main “Calendar” view
def build_calendar_df(weeks_labels=None, overrides=None, moves=None):
    """
    Create a table: rows=modules, cols=weeks; cells list CA names at their deadline week.
    Also annotate preview moves and last-applied moves.
    """
    weeks_labels = weeks_labels or st.session_state.weeks  # column labels
    overrides = overrides or st.session_state.get("preview_overrides", {})  # preview position changes
    moves = moves or st.session_state.get("last_applied_moves", {})         # applied changes (old→new)
    n_weeks = len(weeks_labels)  # number of columns
    table = {mod: [""] * n_weeks for mod in st.session_state.meta.keys()}  # start empty cells

    # Fill CA labels into the right week cells
    for mod, cas_list in st.session_state.ca_map.items():
        name_map = st.session_state.ca_names.get(mod, {})
        for (idx, wt, dl, rl) in cas_list:
            if 1 <= dl <= 12:
                final_dl = overrides.get((mod, idx), dl)   # use override deadline if previewing
                moved_pair = moves.get((mod, idx))         # (old_dl, new_dl) if just applied

                label = name_map.get(idx, f"CA #{idx}")    # show display name if available
                try:
                    pct = f"({float(wt):.0f}%)" if wt not in (None, 0) else ""  # weight like "(20%)"
                except Exception:
                    pct = f"({wt}%)" if wt not in (None, 0) else ""
                base_text = f"{label} {pct}".strip()       # final cell text

                # If this CA was just moved and data still reflects it, show old and new places
                if moved_pair:
                    old_dl, new_dl = moved_pair
                    if dl == new_dl:
                        old_idx = old_dl - 1
                        ghost = f"[moved_from] {base_text} → {new_dl}"  # ghost marker at old cell
                        prev_old = table[mod][old_idx]
                        table[mod][old_idx] = (prev_old + "\n" if prev_old else "") + ghost
                        text = f"{base_text} [{old_dl}→{new_dl}]"       # annotate current cell
                    else:
                        text = base_text  # if undone later, just show plain text
                # If only previewing, show dashed/yellow style via a tag
                elif final_dl != dl:
                    text = f"{base_text} → {final_dl}"  # show arrow to preview week
                    prev_old = table[mod][dl - 1]
                    ghost = f"[preview_from] {base_text} → {final_dl}"  # preview marker at old cell
                    table[mod][dl - 1] = (prev_old + "\n" if prev_old else "") + ghost
                else:
                    text = base_text  # no change

                # Place the label into the final (possibly overridden) deadline column
                col_idx = final_dl - 1
                prev = table[mod][col_idx]
                table[mod][col_idx] = (prev + "\n" if prev else "") + text

    # Add exam info to week 15 if the module has an exam share
    if "week 15" in weeks_labels:
        idx15 = weeks_labels.index("week 15")
        for mod, (_credits, assign_pct, _contact) in st.session_state.meta.items():
            exam_pct = max(0.0, (1.0 - float(assign_pct)) * 100.0)  # exam percentage of total
            if exam_pct > 0:
                prev = table[mod][idx15]
                table[mod][idx15] = (prev + "\n" if prev else "") + f"Exam ({exam_pct:.0f}%)"

    # Convert dict-of-lists to a dataframe and set index name
    df = pd.DataFrame.from_dict(table, orient="index", columns=weeks_labels)
    df.index.name = "Module"
    return df

# Compute weight distribution across a span (release..deadline) for a given style
def _weights_for_span(d: int, style: str):
    """
    Return a list of weights for (d+1) weeks based on study style.
    """
    if d < 0:
        return []
    if style == "Early Starter":
        return [1/(d+1)] * (d+1)  # equal weights
    if style == "Steady":
        denom = (d**2 + 3*d + 2) or 1  # safe denominator
        return [2*(i+1)/denom for i in range(d+1)]  # ramp up towards deadline
    if style == "Just in Time":
        return [0]*d + [1]  # all work at the deadline week
    return [1/(d+1)] * (d+1)  # fallback equal split

# Build weekly hour allocations for every module (15 weeks), optionally with deadline overrides
def recompute_all_weekly(study_style: str,
                         meta: dict,
                         ca_map: dict,
                         deadline_overrides: dict | None = None) -> dict:
    """
    Return dict of module -> list of 15 weekly hours, applying optional deadline overrides.
    """
    deadline_overrides = deadline_overrides or {}  # default: no overrides
    weeks15_by_mod = {}  # result store

    for mod in meta:
        credits, assign_pct, contact = meta[mod]  # unpack module meta
        total_notional = credits * 10             # total hours for the module

        # Start with contact hours in teaching weeks; week 7 = 0; weeks 13..15 start at 0
        weekly = [contact]*6 + [0] + [contact]*5 + [0, 0, 0]

        # Remaining coursework hours to spread across 1..12 (ignoring week 7)
        prep_time = max(total_notional * assign_pct - sum(weekly[:12]), 0.0)

        ca_list = ca_map.get(mod, [])             # CAs for this module
        total_pct = sum(w for (_idx, w, _dl, _rl) in ca_list) or 1.0  # sum of CA weights

        for (idx, wt, dl, rl) in ca_list:
            dl = deadline_overrides.get((mod, idx), dl)  # override deadline if provided

            d = dl - rl  # span length (inclusive count is d+1)
            if d < 0 or rl < 1 or dl > 12:
                continue  # skip invalid spans or deadlines outside 1..12

            T = prep_time * (wt / total_pct)  # hours to allocate for this CA
            weights = _weights_for_span(d, study_style)  # distribution across the span

            # Ensure week 7 is zero and exam weeks are handled separately
            weekly[6] = 0
            weekly[-3:] = [0, 0, 0]

            # Add CA effort into weeks 1..12, skipping week 7
            for i, w in enumerate(weights):
                week_idx = rl - 1 + i
                if 0 <= week_idx < 12 and week_idx != 6:
                    weekly[week_idx] += T * w

        # Distribute exam effort over weeks 13..15
        exam_pct = 1.0 - assign_pct
        exam_effort = total_notional * exam_pct
        d_exam = 2  # span for weeks 13..15
        exam_w = _weights_for_span(d_exam, study_style)
        for i, w in enumerate(exam_w):
            weekly[12 + i] += exam_effort * w

        # Store the computed 15-week profile
        weeks15_by_mod[mod] = weekly

    return weeks15_by_mod

# Compute overall CV% across total hours in weeks 1..12 (ignoring week 7)
def total_cv_percent(weeks15_by_mod: dict) -> float:
    """
    Return CV% for the sum of hours per week over weeks 1..12.
    """
    totals = [0.0]*12  # total hours per week across all modules
    for weekly in weeks15_by_mod.values():
        for i in range(12):
            totals[i] += weekly[i]
    mean = sum(totals)/12 if totals else 0.0
    if mean == 0:
        return 0.0
    var = sum((x - mean)**2 for x in totals) / 12  # population variance
    std = var ** 0.5
    return (std / mean) * 100.0  # CV% = std/mean*100

# Save/replace a module's CA list, update session, and persist to DB
def persist_cas(name, credits, assign_pct, ca_nms, ca_wts, ca_dls, ca_rels):
    """
    Validate CA rows, update state, and write to DB for a module.
    """
    notional  = credits * assign_pct * 10                  # coursework hours
    weekly    = st.session_state.baseline[name].copy()     # start from baseline
    prep_time = max(notional - sum(weekly[:12]), 0.0)      # to allocate across 1..12
    total_pct = sum(ca_wts) or 1.0                         # sum of CA weights

    cas_list = []  # will hold (idx, wt, dl, rel)
    for idx, (wt, dl, rel, nm) in enumerate(zip(ca_wts, ca_dls, ca_rels, ca_nms), start=1):
        if (dl - rel) < 0 or rel < 1 or dl > 12:
            st.warning(f"Invalid release/deadline for CA#{idx} — skipped.")
            continue
        cas_list.append((idx, wt, dl, rel))                # store CA row
        weekly[dl - 1] += prep_time * (wt / total_pct)     # add CA chunk at deadline
        st.session_state.ca_names.setdefault(name, {})[idx] = nm  # save display name

    # Zero week 7 and keep exam separate in 13..15
    weekly[6]   = 0
    weekly[-3:] = [0, 0, 0]

    # Update session state and persist to DB
    st.session_state.modules[name] = weekly
    st.session_state.ca_map[name]  = cas_list
    save_cas(name, cas_list)
    save_schedule(name, weekly)

# Cache wrapper for weekly recomputation to speed up scenario testing
@st.cache_data(show_spinner=False)
def _weeks15_cached(study_style, meta_items, ca_map_items, overrides_items):
    """Cached call to recompute_all_weekly with hashable inputs."""
    meta = dict(meta_items)
    ca_map = {k: list(v) for k, v in ca_map_items}
    overrides = dict(overrides_items) if overrides_items else None
    return recompute_all_weekly(study_style, meta, ca_map, overrides)

# Convert dicts to tuples so they can be used as cache keys
def _hashables_for_cache(meta, ca_map, overrides=None):
    """
    Turn dicts into sorted tuples so they are hashable for caching.
    """
    meta_items = tuple(sorted((k, tuple(v)) for k, v in meta.items()))
    ca_map_items = tuple(sorted((k, tuple(tuple(x) for x in v)) for k, v in ca_map.items()))
    overrides_items = tuple(sorted(overrides.items())) if overrides else None
    return meta_items, ca_map_items, overrides_items

# Try all combinations of exactly Kmax CA shifts (±1 week) and compute CVs
def generate_scenarios_exact_upto_k(all_cas, Kmax, study_style, meta, ca_map, valid_fn):
    """
    Return list of scenarios with exactly Kmax shifts: (k, cv, description).
    """
    scenarios = []
    if not all_cas or Kmax <= 0:
        return scenarios

    # Prepare cacheable inputs once
    meta_items, ca_map_items, _ = _hashables_for_cache(meta, ca_map, None)

    k = Kmax  # we only do “exactly Kmax”

    for idxs in itertools.combinations(range(len(all_cas)), k):
        for dirs in itertools.product([-1, 1], repeat=k):
            overrides, changes, valid = {}, [], True
            for pos, dir_ in zip(idxs, dirs):
                mod, idx, rl, dl = all_cas[pos]
                new_dl = dl + dir_
                if not valid_fn(rl, new_dl):  # skip illegal shifts
                    valid = False
                    break
                overrides[(mod, idx)] = new_dl  # collect override
                changes.append(f"{mod} CA#{idx}@week {dl}→{new_dl}")  # human text
            if not valid:
                continue

            # Use cached recompute for speed
            _, _, overrides_items = _hashables_for_cache(meta, ca_map, overrides)
            weeks15 = _weeks15_cached(study_style, meta_items, ca_map_items, overrides_items)
            cv = total_cv_percent(weeks15)

            scenarios.append((k, cv, "none" if not changes else "; ".join(changes)))
    return scenarios

# If old code set step==2, redirect it to the Results page
if st.session_state.step == 2:
    st.session_state.step = 3
    st.rerun()

# ---------- STEP 0: list of modules (view, add, edit, delete) ----------
if st.session_state.step == 0:
    st.title("All Modules")  # page title
    c1, c2 = st.columns([3,1])  # layout columns
    with c2:
        # Button to start creating a new module
        if st.button("➕ Add New Module"):
            st.session_state.selected = "__new__"  # mark as new
            st.session_state["_name"] = ""         # clear fields
            st.session_state["_credits"] = 0.0
            st.session_state["_assign_pct"] = 0.0
            st.session_state["_contact"] = 0.0
            st.session_state["_n_ca"] = 0
            # Remove any leftover CA form values from earlier edits
            for k in list(st.session_state.keys()):
                if k.startswith(("wt","dl","rel","nm")):
                    del st.session_state[k]
            st.session_state.step = 1  # go to module setup
            st.rerun()

    # Show each module with summary and actions
    for mod in st.session_state.meta:
        with st.expander(mod):
            cr, ap, ct = st.session_state.meta[mod]  # unpack
            exam_pct = 1.0 - ap                      # exam share
            st.markdown(
                f"**Credits:** {cr}  •  "
                f"**CW %:** {ap*100:.0f}%  •  "
                f"**Exam %:** {exam_pct*100:.0f}%  •  "
                f"**Contact:** {ct}h/wk"
            )
            # Edit button opens the setup form prefilled
            if st.button("✏️ Edit", key=f"edit_{mod}"):
                for k in list(st.session_state.keys()):
                    if k.startswith(("wt", "dl", "rel", "nm")) or k in {"_name","_credits","_assign_pct","_contact","_n_ca"}:
                        del st.session_state[k]
                st.session_state.selected = mod
                st.session_state.step = 1
                st.rerun()

            # Delete button removes from DB and session
            if st.button("🗑️ Delete", key=f"delete_{mod}"):
                from db import delete_module
                delete_module(mod)
                # Remove from session_state too
                st.session_state.meta.pop(mod, None)
                st.session_state.ca_map.pop(mod, None)
                st.session_state.modules.pop(mod, None)
                st.session_state.baseline.pop(mod, None)
                st.success(f"Module '{mod}' deleted.")
                st.rerun()

    # Helper text when nothing selected
    st.write("Click ‘Add New Module’ or ‘Edit’ to begin.")

# ---------- STEP 1: module definition and CA entry ----------
elif st.session_state.step == 1:
    st.title("Module Definitions")  # page title

    # If editing an existing module, prefill form fields once
    if st.session_state.selected and st.session_state.selected != "__new__":
        sel = st.session_state.selected
        cr, ap, ct = st.session_state.meta[sel]

        # Basic module fields
        if "_name" not in st.session_state:        st.session_state["_name"] = sel
        if "_credits" not in st.session_state:     st.session_state["_credits"] = cr
        if "_assign_pct" not in st.session_state:  st.session_state["_assign_pct"] = ap * 100
        if "_contact" not in st.session_state:     st.session_state["_contact"] = ct
        if "_n_ca" not in st.session_state:        st.session_state["_n_ca"] = len(st.session_state.ca_map.get(sel, []))

        # CA fields (one set per CA)
        for idx, wt, dl, rel in st.session_state.ca_map.get(sel, []):
            i = idx - 1
            if f"wt{i}"  not in st.session_state: st.session_state[f"wt{i}"]  = wt
            if f"dl{i}"  not in st.session_state: st.session_state[f"dl{i}"]  = dl
            if f"rel{i}" not in st.session_state: st.session_state[f"rel{i}"] = rel
            if f"nm{i}"  not in st.session_state:
                st.session_state[f"nm{i}"] = st.session_state.ca_names.get(sel, {}).get(idx, "")

    # Build the module form
    with st.form("module_form"):
        name       = st.text_input("Module name", value=st.session_state.get("_name",""))  # module name
        credits    = st.number_input("Credits", min_value=0.0, step=0.5,
                                     value=st.session_state.get("_credits",0.0))          # credits value
        # This is kept for backward compatibility, but we soon mirror it to CW%
        assign_pct = st.number_input("Assignment % of total hours",
                                     min_value=0.0, max_value=100.0,
                                     value=st.session_state.get("_assign_pct",0.0))/100.0
        
        # Two paired inputs to show CW% and derived Exam%
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
        assign_pct = cw_pct  # store back the CW% as the assignment proportion

        contact    = st.number_input("Contact hrs/week",
                                     min_value=0.0, step=0.5,
                                     value=st.session_state.get("_contact",0.0))  # teaching hours per week

        n_ca = st.number_input("How many CAs?", min_value=0, step=1,
                               value=st.session_state.get("_n_ca",0))  # number of CA rows
        
        # Lists to collect CA inputs from the form
        ca_nms, ca_wts, ca_dls, ca_rels = [], [], [], []

        # If there are CAs, draw the inputs for each
        if n_ca > 0:
            st.subheader("Continuous Assessments")
            for i in range(int(n_ca)):
                nm = st.text_input(  # CA display name
                    f"Name for CA #{i+1}",
                    key=f"nm{i}",
                    value=st.session_state.get(f"nm{i}", "")
                )
                c1, c2, c3 = st.columns([2, 1, 1])  # weight, deadline, release
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
                # Push values to lists
                ca_nms.append(nm); ca_wts.append(w); ca_dls.append(d); ca_rels.append(r)

        # Three form buttons: back, save/update, allocate CAs
        b1, b2, b3 = st.columns(3)
        with b1: back = st.form_submit_button("◀ Back to list")
        with b2: save_mod = st.form_submit_button("Save / Update Module")
        with b3: alloc_cas = st.form_submit_button("Allocate CAs")

    # Ensure booleans exist after form submit (Streamlit rerun safety)
    back = bool(back) if 'back' in locals() else False
    save_mod = bool(save_mod) if 'save_mod' in locals() else False
    alloc_cas = bool(alloc_cas) if 'alloc_cas' in locals() else False

    # Back button: return to list without saving anything
    if back:
        go_home()
        st.stop()

    # Save button: persist module meta and optional CA rows
    if save_mod:
        if not name:
            st.error("Module name is required.")
        else:
            # Build baseline schedule with contact hours and zeros where needed
            sched = [contact]*6 + [0] + [contact]*5 + [0, 0, 0]

            # Update session with baseline and meta
            st.session_state.baseline[name] = sched.copy()
            st.session_state.modules[name]  = sched.copy()
            st.session_state.meta[name]     = (credits, assign_pct, contact)

            # Write meta and schedule to DB
            save_module_meta(name, credits, assign_pct, contact)
            save_schedule(name, sched)

            # If CA inputs exist, persist them too
            if int(n_ca) > 0:
                persist_cas(name, credits, assign_pct, ca_nms, ca_wts, ca_dls, ca_rels)

            # Notify and go back to module list
            st.success(f"Module '{name}' saved.")
            st.session_state.selected = None
            st.session_state.step = 0
            st.rerun()

    # Allocate CAs button: overwrite CAs for an already-saved module
    if alloc_cas:
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

# ---------- STEP 3: results (workload table, scenarios, calendar, chart) ----------
elif st.session_state.step == 3:

    st.title("Results")  # page title
    weeks = st.session_state.weeks  # all 15 week labels
    teaching_weeks = weeks[:12]     # consider 1..12 for CV calculation

    # Dropdown to choose the study style used for allocation
    study_style = st.selectbox(
        "Study Style",
        ["Early Starter", "Steady", "Just in Time"],
        index=["Early Starter", "Steady", "Just in Time"].index(
            st.session_state.get("study_style", "Just in Time")
        )
    )
    st.session_state["study_style"] = study_style  # remember choice
    st.caption("This controls how effort is distributed from CA release to deadline during allocation.")

    # Recompute weekly workloads per module using the selected study style
    df_rows = {}
    for mod in st.session_state.meta:
        credits, assign_pct, contact = st.session_state.meta[mod]
        baseline = [contact]*6 + [0] + [contact]*5 + [0, 0, 0]  # base teaching pattern
        weekly = baseline.copy()                                # mutable list
        total_notional = credits * 10                           # total hours
        prep_time = max(total_notional * assign_pct - sum(weekly[:12]), 0.0)  # CA hours to allocate

        ca_list = st.session_state.ca_map.get(mod, [])          # CA definitions
        total_pct = sum(w for (_, w, _, _) in ca_list) or 1.0   # sum CA weights

        for idx, wt, dl, rel in ca_list:
            T = prep_time * (wt / total_pct)                    # hours for this CA
            d = dl - rel                                        # span
            if d < 0 or rel < 1 or dl > 12:
                continue

            # Pick weights by style
            if study_style == "Early Starter":
                weights = [1 / (d + 1)] * (d + 1)
            elif study_style == "Steady":
                denom = (d**2 + 3*d + 2) or 1
                weights = [2 * (i + 1) / denom for i in range(d + 1)]
            else:
                weights = [0] * d + [1]

            weekly[6] = 0            # week 7 off
            weekly[-3:] = [0, 0, 0]  # exams handled later

            # Add CA effort to weeks 1..12 (skip week 7)
            for i, w in enumerate(weights):
                week_idx = rel - 1 + i
                if 0 <= week_idx < 12 and week_idx != 6:
                    weekly[week_idx] += T * w

        # Spread exam effort (1 - assignment%) over weeks 13..15
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

        # Save module row
        df_rows[mod] = weekly

    # Build DataFrame with rows=modules (+TOTAL row) and columns=weeks 1..15
    df_main = pd.DataFrame(df_rows, index=weeks).T
    tot      = df_main[weeks].sum(axis=0)                         # sum per week
    df_total = pd.DataFrame([tot.values], index=["TOTAL"], columns=weeks)  # TOTAL row
    df       = pd.concat([df_main, df_total], axis=0)

    # Add per-row totals and CV% (Pain Score) for weeks 1..12
    df["Total"] = df[teaching_weeks].sum(axis=1)
    df["Pain Score"]    = (df[teaching_weeks].std(ddof=0, axis=1) / df[teaching_weeks].mean(axis=1) * 100).round(1)
    df["Colour"]= df["Pain Score"].map(colour_for_cv)

    # Show/hide toggle for the table to keep the page lighter
    if "show_workload" not in st.session_state:
        st.session_state.show_workload = False

    col_show, _ = st.columns([1, 7])
    if col_show.button(("Hide" if st.session_state.show_workload else "Show") + " workload table"):
        st.session_state.show_workload = not st.session_state.show_workload

    # If showing, render a styled HTML table; else keep df for Excel
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
        df_disp = df.round(1)

    # Helper to validate a shift (new deadline is within 1..12 and >= release)
    def _valid_shift(rl: int, new_dl: int) -> bool:
        return 1 <= new_dl <= 12 and new_dl >= rl

    # Build a list of all CAs that can be shifted (deadline within 1..12)
    _all_cas = []
    for _m, _cas_list in st.session_state.ca_map.items():
        for (_i, _w, _dl, _rl) in _cas_list:
            if 1 <= _dl <= 12:
                _all_cas.append((_m, _i, _rl, _dl))

    # Limit Excel scenario enumeration to up to 3 shifts
    _excel_Kmax = min(3, len(_all_cas))
    _scenarios = generate_scenarios_exact_upto_k(
        _all_cas, _excel_Kmax, study_style,
        st.session_state.meta, st.session_state.ca_map, _valid_shift
    )

    # Prepare “All Scenarios” sheet data (with columns kept as originally used elsewhere)
    out = (pd.DataFrame(_scenarios, columns=["no_shifts","CV","changes"])
           if _scenarios else pd.DataFrame(columns=["no_shifts","CV","changes"]))
    out = out.sort_values(["CV","no_shifts"], ascending=[True,True]).reset_index(drop=True)

    # Button that jumps to the heatmaps page for all module pairs
    if st.button("Show all module-pair heatmaps"):
        st.session_state.heatmap_modules = "__ALL__"
        st.session_state.step = "HEATMAP"
        st.rerun()

    # Calendar section heading
    st.markdown("### Calendar")

    # Compute baseline CV for current data
    base_weeks15 = recompute_all_weekly(study_style, st.session_state.meta, st.session_state.ca_map)
    base_cv = total_cv_percent(base_weeks15)

    # If there is a preview set of overrides, compute its CV
    preview_overrides = st.session_state.get("preview_overrides", None)
    preview_cv = None
    if preview_overrides:
        prev_weeks = recompute_all_weekly(study_style, st.session_state.meta, st.session_state.ca_map, preview_overrides)
        preview_cv = total_cv_percent(prev_weeks)

    # Check if we just applied a shift and want to show before/after once
    _last = st.session_state.pop("_last_cv_change", None)

    # Build badges row: show before/after if available, else current; also preview
    badges = []
    if _last:
        before, after = _last
        badges.append(cv_badge(before, "Before"))
        badges.append(cv_badge(after,  "After"))
    else:
        badges.append(cv_badge(base_cv, "Current"))

    if preview_cv is not None:
        badges.append(cv_badge(preview_cv, "Preview"))

    # Render badges (HTML)
    st.markdown("".join(badges), unsafe_allow_html=True)

    # Small caption below badges describing current Pain score and delta if any
    if _last:
        before, after = _last
        arrow = "↓" if after < before else ("↑" if after > before else "→")
        st.markdown(
            f"<p class='caption-lg'>Pain score (weeks 1–12): "
            f"<b>{after:.1f}%</b> <span style='opacity:.8'>({arrow} from {before:.1f}%)</span></p>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<p class='caption-lg'>Pain score (weeks 1–12): <b>{base_cv:.1f}%</b></p>",
            unsafe_allow_html=True
        )

    # Build the calendar dataframe and a display copy with Module as a column
    cal_df = build_calendar_df(st.session_state.weeks)
    cal_df_display = cal_df.reset_index()
    week_cols = [c for c in cal_df_display.columns if c.startswith("week")]  # week columns

    n_weeks = len(week_cols)  # number of week columns
    module_pct = 16           # % width of Module column
    week_pct   = (100 - module_pct) / n_weeks  # width per week column

    # Inject CSS to make the calendar table scrollable and readable
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

    # If preview overrides exist, use them for the per-module stripe coloring
    preview = st.session_state.get("preview_overrides", {})
    weeks15_by_mod = recompute_all_weekly(study_style, st.session_state.meta, st.session_state.ca_map,
                                          preview if preview else None)

    # Helper to compute per-module CV% (1..12) used for the left stripe color
    def _cv_percent_1to12(weekly):
        vals = weekly[:12]
        m = sum(vals)/12 if vals else 0.0
        if m == 0: return 0.0
        var = sum((x - m)**2 for x in vals) / 12
        return (var ** 0.5) / m * 100.0

    # Map color names to hex for the stripe
    band_hex = {"green":"#2ecc71","yellow":"#f1c40f","orange":"#e67e22","red":"#e74c3c","black":"#000000"}
    # Decide stripe color per module based on its CV band
    row_band = {m: colour_for_cv(_cv_percent_1to12(w)) for m,w in weeks15_by_mod.items()}

    # Style function for week columns (adds backgrounds, outlines, stripes)
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

    # Style function for Module column to add a wider left stripe
    def _stripe_module_col(col: pd.Series) -> list[str]:
        styles = []
        for i, _ in enumerate(col):
            module = cal_df_display.iloc[i]["Module"]
            stripe = band_hex.get(row_band.get(module, "green"), "#2ecc71")
            styles.append(f"border-left: 8px solid {stripe};")
        return styles

    # Imports needed to generate a cache key for the calendar HTML
    import hashlib, json

    # Create a stable key for caching the HTML representation of the calendar
    def _calendar_key(df_display, row_band, week_cols):
        # Get preview/applied info from session (or empty)
        preview_raw = st.session_state.get("preview_overrides", {}) or {}
        moves_raw   = st.session_state.get("last_applied_moves", {}) or {}

        # Convert tuple keys to strings so we can JSON-serialize
        preview_safe = {f"{m}::{i}": dl for (m, i), dl in preview_raw.items()}
        # Convert move tuples to lists for JSON
        moves_safe   = {f"{m}::{i}": [old, new] for (m, i), (old, new) in moves_raw.items()}

        # Build a compact JSON payload that changes when the table visuals change
        payload = {
            "data": df_display.fillna("").astype(str).to_dict(orient="list"),
            "row_band": row_band,
            "preview": preview_safe,
            "moves": moves_safe,
            "weeks": week_cols,
        }
        s = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha1(s).hexdigest()

    # Prepare a place in session to cache the calendar HTML
    if "calendar_cache" not in st.session_state:
        st.session_state.calendar_cache = {}

    # Compute the cache key for the current calendar state
    cal_key = _calendar_key(cal_df_display, row_band, week_cols)
    html = st.session_state.calendar_cache.get(cal_key)

    # If not cached, build the styled HTML and store it
    if html is None:
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
        html = calendar_styled.to_html()
        st.session_state.calendar_cache[cal_key] = html

    # Render the calendar inside a scrollable container
    st.markdown(f"<div class='calendar-wrap'>{html}</div>", unsafe_allow_html=True)

    # Short legend explaining the colors/styles used in the calendar cells
    st.caption("Legend: left border colour = CV band (green/yellow/orange/red/black); pale blue = CA; deeper amber = multiple CAs; yellow (dashed) = preview move; purple = exam; grey = week 7.")

    # Add a visible legend with the same colors as used in the cells
    st.markdown("""
    <div style="margin:8px 0; display:flex; flex-wrap:wrap; gap:12px;">

    <span style="display:inline-block; background-color:#2ecc71; color:#fff; padding:4px 10px; border-radius:4px;">
        CV band: Green (&lt;92%)
    </span>

    <span style="display:inline-block; background-color:#f1c40f; color:#111; padding:4px 10px; border-radius:4px;">
        CV band: Yellow (92–101.9%)
    </span>

    <span style="display:inline-block; background-color:#e67e22; color:#fff; padding:4px 10px; border-radius:4px;">
        CV band: Orange (102–111.9%)
    </span>

    <span style="display:inline-block; background-color:#e74c3c; color:#fff; padding:4px 10px; border-radius:4px;">
        CV band: Red (112–119.9%)
    </span>

    <span style="display:inline-block; background-color:#000; color:#fff; padding:4px 10px; border-radius:4px;">
        CV band: Black (≥120%)
    </span>

    <span style="display:inline-block; background-color:#e3f2fd; color:#111; padding:4px 10px; border-radius:4px;">
        CA deadline
    </span>

    <span style="display:inline-block; background-color:#ffcc80; color:#111; padding:4px 10px; border-radius:4px;">
        Multiple CAs
    </span>

    <span style="display:inline-block; background-color:#fff3cd; color:#111; border:2px dashed #856404; padding:4px 10px; border-radius:4px;">
        Preview move
    </span>

    <span style="display:inline-block; background-color:#ede7f6; color:#111; padding:4px 10px; border-radius:4px;">
        Exam
    </span>

    <span style="display:inline-block; background-color:#3a3a3a; color:#eee; padding:4px 10px; border-radius:4px;">
        Week 7 (Reading week)
    </span>

    </div>
    """, unsafe_allow_html=True)

    # Recommendations header
    st.subheader("Recommendations")

    # Default number of shifts to evaluate
    if "N" not in st.session_state:
        st.session_state.N = 2

    # UI to set N and show results
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

    # If we have an undo available, show the undo box
    if st.session_state.get("undo_payload"):
        with st.container(border=True):
            st.write("**Undo last shift**")
            if st.session_state.get("undo_caption"):
                st.caption(st.session_state.undo_caption)
            if st.button("Undo"):
                # Revert each CA deadline to its previous value
                for (m, idx), old_dl in st.session_state.undo_payload.items():
                    old_list = st.session_state.ca_map.get(m, [])
                    new_list = []
                    for (j, wt, dl, rl) in old_list:
                        if j == idx:
                            dl = old_dl
                        new_list.append((j, wt, dl, rl))
                    st.session_state.ca_map[m] = new_list
                    save_cas(m, new_list)
                # Clear last moves and undo info
                st.session_state.last_applied_moves = {}
                st.session_state.undo_payload = None
                st.session_state.undo_caption = ""
                st.success("Reverted last shift.")
                st.rerun()

    # Build the list of adjustable CAs again (deadline within 1..12)
    all_cas = []
    for mod, cas_list in st.session_state.ca_map.items():
        for (idx, wt, dl, rl) in cas_list:
            if 1 <= dl <= 12:
                all_cas.append((mod, idx, rl, dl))

    # Generate scenarios that use exactly N shifts
    scenarios = generate_scenarios_exact_upto_k(
        all_cas, st.session_state.N, study_style,
        st.session_state.meta, st.session_state.ca_map, _valid_shift
    )

    # Repeat header for clarity (kept as-is)
    st.subheader("Recommendations")

    # If we found scenarios, build a table and show top options
    if scenarios:
        df_scen = pd.DataFrame(scenarios, columns=["no_shifts","Pain score","changes"])
        filtered = df_scen.query("no_shifts == @st.session_state.N")

        out_visible = (
            filtered
            .sort_values(["Pain score","no_shifts"], ascending=[True, True])
            .reset_index(drop=True)
        )

        topN = out_visible.head(5).copy()

        # For each top scenario, show a preview and an apply button
        for i, row in topN.iterrows():
            with st.container(border=True):
                st.write(f"**Option {i+1}** — Pain score **{row['Pain score']:.1f}%**")
                st.caption(row["changes"] if row["changes"] != "none" else "No changes")
                c1, c2, _ = st.columns([1,1,6])

                # Unique ID for buttons to avoid collisions
                uniq = f"{i}_{abs(hash(row['changes'])) % 10_000_000}"

                # Visualize button: store overrides and rerun (yellow dashed calendar)
                if c1.button("Visualize", key=f"viz_cal_{uniq}"):
                    st.session_state["preview_overrides"] = {
                        (p.split(' CA#')[0].strip(), int(p.split(' CA#')[1].split('@')[0])):
                        int(p.split('→')[1].strip())
                        for p in [x.strip() for x in row["changes"].split(';')] if p and "→" in p
                    }
                    st.rerun()

                # Shift now button: apply the overrides, persist, enable undo, and recalc CV
                if c2.button("Shift now", key=f"apply_cal_{uniq}"):
                    overrides = {
                        (p.split(' CA#')[0].strip(), int(p.split(' CA#')[1].split('@')[0])):
                        int(p.split('→')[1].strip())
                        for p in [x.strip() for x in row["changes"].split(';')] if p and "→" in p
                    }
                    if not overrides:
                        st.info("No changes to apply.")
                    else:
                        # CV before applying changes
                        cv_before = total_cv_percent(
                            recompute_all_weekly(study_style, st.session_state.meta, st.session_state.ca_map)
                        )

                        # Build undo map and record last moves
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

                        # Apply new deadlines to CA map and save to DB
                        for (m, idx), new_dl in overrides.items():
                            old_list = st.session_state.ca_map.get(m, [])
                            new_list = []
                            for (j, wt, dl, rl) in old_list:
                                if j == idx:
                                    dl = new_dl
                                new_list.append((j, wt, dl, rl))
                            st.session_state.ca_map[m] = new_list
                            save_cas(m, new_list)

                        # Clear preview flag if it was set
                        st.session_state.pop("preview_overrides", None)

                        # CV after applying changes
                        cv_after = total_cv_percent(
                            recompute_all_weekly(study_style, st.session_state.meta, st.session_state.ca_map)
                        )

                        # Store before/after to show once in badges
                        st.session_state["_last_cv_change"] = (cv_before, cv_after)

                        # Inform and rerun to refresh visuals
                        st.success(f"Shifts applied. New Pain score: {cv_after:.1f}% (was {cv_before:.1f}%).")
                        st.rerun()

    else:
        # No scenarios available
        st.info("No candidate scenarios available with ±1 week shifts.")

    # If we are in a preview state, allow clearing it
    if st.session_state.get("preview_overrides"):
        if st.button("Reset preview"):
            st.session_state.pop("preview_overrides", None)
            st.rerun()

    # Build an Excel file with two sheets (Workload + All Scenarios)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_disp.to_excel(writer, sheet_name="Workload")
        out.to_excel(writer, sheet_name="All Scenarios", index=False)
    # Download button for the Excel export
    st.download_button(
        " Download Excel",
        data=buf.getvalue(),
        file_name="results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # Prepare dataframe for the line chart of workloads over time
    chart_df = (
        df[weeks]
        .reset_index()
        .melt(id_vars="index", var_name="Week", value_name="Hours")
        .rename(columns={"index":"Module"})
    )
    # Section header for the chart
    st.subheader("Workload Over Time")
    # Create a line chart with a thicker line for TOTAL
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
    # Render the chart
    st.altair_chart(chart, use_container_width=True)

# ---------- STEP "HEATMAP": all-pairs heatmaps (and single-pair fallback) ----------
elif st.session_state.step == "HEATMAP":
    flag = st.session_state.get("heatmap_modules")                 # which pair(s) to show
    style = st.session_state.get("study_style", "Just in Time")    # current study style

    # If user chose the view for all pairs, build all grids
    if flag == "__ALL__":

        # Helper: check if module has any CA deadline in 1..12
        def _has_teaching_ca(mod):
            return any(1 <= dl <= 12 for (_i, _w, dl, _r) in st.session_state.ca_map.get(mod, []))

        # Filter modules with at least one CA in teaching weeks
        mods_with_cas = [m for m in st.session_state.meta.keys() if _has_teaching_ca(m)]
        # Build all unordered pairs of such modules
        pairs = list(itertools.combinations(mods_with_cas, 2))

        # Page title
        st.title("All module-pair heatmaps")

        # Function to compute and/or render the 12x12 lattice for a pair
        def render_pair(A_mod, B_mod, highlight_coords=None, compute_only=False):
            # Only show a heading if we are rendering (not in compute-only pass)
            if not compute_only:
                st.markdown(f"### {A_mod} vs {B_mod} — CV cross heatmap")

            # Get CA rows for both modules limited to teaching weeks
            A_cas = [(idx, wt, dl, rl) for (idx, wt, dl, rl) in st.session_state.ca_map.get(A_mod, []) if 1 <= dl <= 12]
            B_cas = [(idx, wt, dl, rl) for (idx, wt, dl, rl) in st.session_state.ca_map.get(B_mod, []) if 1 <= dl <= 12]

            # Which weeks have any CA in each module (for faint row/col shading)
            A_weeks_with_ca = {dl for (_idx, _wt, dl, _rl) in A_cas}
            B_weeks_with_ca = {dl for (_idx, _wt, dl, _rl) in B_cas}

            # If either module has no CAs in 1..12, skip
            if not A_cas or not B_cas:
                if compute_only:
                    return pd.DataFrame(), None
                st.info("One or both selected modules have no CAs with deadlines in weeks 1–12.")
                return

            # Helper: map CV to band/color and return both
            def cv_band_and_color(cv: float):
                if cv < 92:            return "<92",      "#2ecc71"
                elif cv < 102:         return "92–101.9", "#f1c40f"
                elif cv < 112:         return "102–111.9","#e67e22"
                elif cv < 120:         return "112–119.9","#e74c3c"
                else:                  return "≥120",     "#b71c1c"

            # Helper: check whether a shift for a given module/week by dir_ is legal
            def can_shift(mod: str, w: int, dir_: int) -> bool:
                new_w = w + dir_
                if not (1 <= new_w <= 12):
                    return False
                for (_idx, _wt, dl, rl) in st.session_state.ca_map.get(mod, []):
                    if dl == w and new_w < rl:
                        return False
                return True

            # Collision weeks are those where both modules have the same deadline week
            A_dead = {dl for (_i, _w, dl, _r) in A_cas} if A_cas else set()
            B_dead = {dl for (_i, _w, dl, _r) in B_cas} if B_cas else set()
            collisions = sorted(A_dead & B_dead)

            # Center CV is the current system CV without shifting any deadlines
            base_weeks = recompute_all_weekly(style, st.session_state.meta, st.session_state.ca_map)
            center_cv = total_cv_percent(base_weeks)

            # Prepare a 12x12 grid dict keyed by (A_week, B_week)
            grid = {(a, b): {"cv": None, "count": 0, "label": "", "band": ""} for a in range(1, 13) for b in range(1, 13)}

            # Mark a cell invalid with a dash if we can't compute a value for it
            def set_invalid(a, b):
                cell = grid[(a, b)]
                if cell["label"] == "":
                    cell["label"] = "–"

            # Store a CV value into a cell, keeping the minimum if seen multiple times
            def put_cv(a, b, cv_val: float):
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

            # For each collision week, compute the neighborhood (center, ±1, diagonals, ±2 orthogonals)
            for w in collisions:
                put_cv(w, w, center_cv)  # center cell is the current CV

                # Move A only (±1)
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

                # Move B only (±1)
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

                # Move both (diagonals: ±1,±1)
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

                # Ring-2 orthogonals (A±2,B same) and (B±2,A same)
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

            # Build a tabular dataset from the grid for Altair
            all_cells = [{"A_week": a, "B_week": b} for a in range(1, 13) for b in range(1, 13)]
            rows = []
            for (a, b) in [(r["A_week"], r["B_week"]) for r in all_cells]:
                cell = grid[(a, b)]
                label = cell["label"]
                band  = cell["band"]
                if band == "<92":         color = "#2ecc71"
                elif band == "92–101.9":  color = "#f1c40f"
                elif band == "102–111.9": color = "#e67e22"
                elif band == "112–119.9": color = "#e74c3c"
                elif band == "≥120":      color = "#b71c1c"
                else:                     color = None
                rows.append({
                    "A_week": a, "B_week": b, "Label": label, "Band": band, "Color": color,
                    "IsDash": (label == "–"), "HasBand": band != "",
                    "RowHasA": (a in A_weeks_with_ca), "ColHasB": (b in B_weeks_with_ca),
                })

            grid_df = pd.DataFrame(rows)  # dataframe for charting
            grid_df["CVnum"] = pd.to_numeric(grid_df["Label"], errors="coerce")  # numeric CV
            grid_df["IsCollision"] = grid_df.apply(
                lambda r: (int(r["A_week"]) == int(r["B_week"])) and (int(r["A_week"]) in collisions),
                axis=1
            )

            # Find the minimum CV in this pair (for overall comparison)
            pair_min_row = grid_df.dropna(subset=["CVnum"]).nsmallest(1, "CVnum")
            pair_min_cv = float(pair_min_row.iloc[0]["CVnum"]) if not pair_min_row.empty else None

            # If we are only computing stats, return now
            if compute_only:
                return grid_df, pair_min_cv

            # If global highlight coordinates are provided, mark them
            if highlight_coords:
                highlight_set = set(highlight_coords)
                grid_df["IsGlobalBest"] = grid_df.apply(
                    lambda r: (int(r["A_week"]), int(r["B_week"])) in highlight_set, axis=1
                )
            else:
                grid_df["IsGlobalBest"] = False

            # Define band colors and domains for the heat layer
            band_domain = ["<92", "92–101.9", "102–111.9", "112–119.9", "≥120"]
            band_colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#b71c1c"]

            # Determine chart pixel size (12x12 grid cells)
            cell_size = 60
            chart_width = cell_size * 12
            chart_height = cell_size * 12
            Y_DOMAIN_DESC = list(range(12, 0, -1))  # invert y so 12 is at top

            # Blue-ish shading for rows that have any CA in A module
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

            # Orange-ish shading for columns that have any CA in B module
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

            # Base lattice with visible axes/ticks
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

            # Heat layer with banded colors
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

            # Put the numeric CV (or “–”) in each visible cell
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

            # Blue rectangle to mark collision cells on the diagonal
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

            # Black rectangle to mark globally best CV cells (provided by caller)
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

            # Top axis duplicate (for labels above the grid)
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

            # Right axis duplicate (for labels on the right)
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

            # Combine all layers into one chart
            chart = (lattice + heat + row_shade + col_shade + text_layer + top_axis + right_axis + collision_overlay + best_overlay).properties(
                width=chart_width, height=chart_height
            )

            # Build a small legend explaining the border styles beside the chart
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

            # Place main chart and legend side by side
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
            # Render the chart
            st.altair_chart(full_chart, use_container_width=False)

            # If there were no collisions, let the user know
            if not collisions:
                st.caption("No colliding deadlines for the selected modules (weeks 1–12).")

        # If we have no qualifying pairs, say so
        if not pairs:
            st.info("No module pairs with CAs in weeks 1–12 were found.")
        else:
            # First pass: compute each pair's min CV without rendering (to find global best)
            pair_min_cv_map = {}
            for A_mod, B_mod in pairs:
                grid_df, pair_min_cv = render_pair(A_mod, B_mod, compute_only=True)
                pair_min_cv_map[(A_mod, B_mod)] = pair_min_cv

            # Find the smallest CV among all pairs (if any)
            valid_mins = [v for v in pair_min_cv_map.values() if v is not None]
            global_min_cv = min(valid_mins) if valid_mins else None

            # Show a summary of the global best or warn if none
            if global_min_cv is None:
                st.warning("No numeric CV values found to highlight.")
            else:
                st.success(f"🌟 Global best Pain score is **{global_min_cv:.1f}%** (highlighted in all heatmaps below).")

            # Second pass: render each pair and highlight cells that match the global best
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

        # Back button to return to Results
        if st.button("◀ Previous"):
            st.session_state.step = 3
            st.rerun()

        # Stop further execution after rendering all pairs
        st.stop()

    # --- Fallback for a single pair (stubbed) ---
    pair = st.session_state.get("heatmap_modules")
    if not pair or len(pair) != 2:
        st.error("No module pair selected.")
    else:
        A_mod, B_mod = pair

        st.markdown(f"### {A_mod} vs {B_mod} — Pain score cross heatmap")
        A_cas = [(idx, wt, dl, rl) for (idx, wt, dl, rl) in st.session_state.ca_map.get(A_mod, []) if 1 <= dl <= 12]
        B_cas = [(idx, wt, dl, rl) for (idx, wt, dl, rl) in st.session_state.ca_map.get(B_mod, []) if 1 <= dl <= 12]

        # Sets used to mark rows/cols that have any CA
        A_weeks_with_ca = {dl for (_idx, _wt, dl, _rl) in A_cas}
        B_weeks_with_ca = {dl for (_idx, _wt, dl, _rl) in B_cas}

        # Placeholder area (full single-pair rendering lives above)

        # If no collisions exist, tell the user
        if not collisions:
            st.caption("No colliding deadlines for the selected modules (weeks 1–12).")

    # Back button to return to Results from the single-pair stub
    if st.button("◀ Previous"):
        st.session_state.step = 3
        st.rerun()
