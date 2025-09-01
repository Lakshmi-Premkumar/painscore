
import streamlit as st
import pandas as pd
import itertools
import io
import altair as alt
from streamlit import column_config
import math
import os


from db import ensure_schema, load_all, save_module_meta, save_cas, save_schedule
st.set_page_config(layout="wide")

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


# Create tables only if they don't exist, then load
ensure_schema()
modules_meta, cas_map_db, schedules_db = load_all()


# ─── Session State Setup ──────────────────────────────────────
if 'step'     not in st.session_state: st.session_state.step     = 3
if 'modules'  not in st.session_state: st.session_state.modules  = schedules_db.copy()
if 'baseline' not in st.session_state: st.session_state.baseline = schedules_db.copy()
if 'meta'     not in st.session_state: st.session_state.meta     = modules_meta.copy()
if 'ca_map'   not in st.session_state: st.session_state.ca_map   = cas_map_db.copy()
if 'ca_names' not in st.session_state: st.session_state.ca_names = {}   # {module: {idx: name}}
if 'weeks'    not in st.session_state:
    st.session_state.weeks = [f"week {i}" for i in range(1,16)]
if 'selected' not in st.session_state:
    st.session_state.selected = None
if 'selected_modules' not in st.session_state:
    st.session_state.selected_modules = []          

if 'selected_ca_map' not in st.session_state:
    st.session_state.selected_ca_map = {}          
if 'heatmap_modules' not in st.session_state:
    st.session_state.heatmap_modules = None
if 'last_applied_moves' not in st.session_state:
    # {(module, ca_index): (old_deadline, new_deadline)} for the most recent “Shift now”
    st.session_state.last_applied_moves = {}

# ▼▼ add these 2 lines ▼▼
if 'undo_payload' not in st.session_state:
    st.session_state.undo_payload = None   # {(module, ca_index): old_deadline}
if 'undo_caption' not in st.session_state:
    st.session_state.undo_caption = ""     # human-readable description of last applied change

# ─── Sidebar navigation ──────────────────────────────────────
with st.sidebar:
    st.header("Navigate")
    choice = st.radio(
        "Go to",
        ["Results", "Modules", "Module setup", "Heatmaps (all pairs)"],
        index=0,  # default to Results
        label_visibility="collapsed",
        key="nav_choice"
    )

# Map sidebar choice to the existing step values you already use
if choice == "Results":
    st.session_state.step = 3
elif choice == "Modules":
    st.session_state.step = 0
elif choice == "Module setup":
    st.session_state.step = 1
elif choice == "Heatmaps (all pairs)":
    st.session_state.heatmap_modules = "__ALL__"
    st.session_state.step = "HEATMAP"

   
# ─── Helpers ──────────────────────────────────────────────────
def colour_for_cv(cv: float) -> str:
   
    if cv < 30:         return "green"   
    elif cv < 40:       return "yellow"  
    elif cv < 50:      return "orange"  
    elif cv < 65:      return "red"   
    else:              return "black"  


def normalize_cv(ser):
    raw = ser.std(ddof=0) / ser.mean()
    return raw / (1 + raw)

def go_next():
    st.session_state.step += 1
def go_prev():
    st.session_state.step -= 1
def go_home():
    st.session_state.step = 0
    st.session_state.selected = None
def ca_df(mod):
    """Return a small dataframe of all CAs for a module."""
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
    """Set of deadline weeks (1–12) for a module."""
    cas = st.session_state.ca_map.get(mod, [])
    return {dl for (_idx, _wt, dl, _rl) in cas if 1 <= dl <= 12}
# ---------- Heatmap helpers ----------
def build_calendar_df(weeks_labels=None, overrides=None, moves=None):
    """
    Rows = modules; Cols = weeks 1..15.
    Each cell lists CA name (or 'CA #i') + (weight%) at the *deadline* week.
    If overrides are given (or session has preview_overrides), move those CAs
    to the new week and append '→ new_week' to the label.
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
                # preview move (yellow dashed) still works via overrides
                final_dl = overrides.get((mod, idx), dl)

                # applied move (persisted) — use last_applied_moves for extra hints
                moved_pair = moves.get((mod, idx))  # (old_dl, new_dl) if this CA was moved in the last apply

                label = name_map.get(idx, f"CA #{idx}")
                try:
                    pct = f"({float(wt):.0f}%)" if wt not in (None, 0) else ""
                except Exception:
                    pct = f"({wt}%)" if wt not in (None, 0) else ""
                base_text = f"{label} {pct}".strip()

                # If we have a recorded move, only draw it as "applied" if ca_map still matches the new deadline.
                # (After Undo, dl == old_dl, so we ignore the stale overlay.)
                if moved_pair:
                    old_dl, new_dl = moved_pair
                    if dl == new_dl:
                        # ghost in the old cell
                        old_idx = old_dl - 1
                        ghost = f"[moved_from] {base_text} → {new_dl}"
                        prev_old = table[mod][old_idx]
                        table[mod][old_idx] = (prev_old + "\n" if prev_old else "") + ghost

                        # annotate the current (new) cell
                        text = f"{base_text} [{old_dl}→{new_dl}]"
                    else:
                        # stale record (e.g., after Undo) — no special styling
                        text = base_text


                # Else, if only previewing, show the preview arrow and a ghost in the old cell
                elif final_dl != dl:
                    text = f"{base_text} → {final_dl}"
                    prev_old = table[mod][dl - 1]
                    ghost = f"[preview_from] {base_text} → {final_dl}"
                    table[mod][dl - 1] = (prev_old + "\n" if prev_old else "") + ghost
                else:
                    text = base_text

                # write the main/new cell
                col_idx = final_dl - 1
                prev = table[mod][col_idx]
                table[mod][col_idx] = (prev + "\n" if prev else "") + text


    # Exam label in week 15 (if present)
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
    """Distribution weights from release..deadline inclusive."""
    if d < 0:
        return []
    if style == "Early Starter":
        return [1/(d+1)] * (d+1)
    if style == "Steady":
        denom = (d**2 + 3*d + 2) or 1
        return [2*(i+1)/denom for i in range(d+1)]
    if style == "Just in Time":
        return [0]*d + [1]
    # fallback
    return [1/(d+1)] * (d+1)

def recompute_all_weekly(study_style: str,
                         meta: dict,
                         ca_map: dict,
                         deadline_overrides: dict | None = None) -> dict:
    """
    Build weekly (15 weeks) hours per module, applying optional deadline overrides.
    deadline_overrides keys are (module_name, ca_index) -> new_deadline (int).
    """
    deadline_overrides = deadline_overrides or {}
    weeks15_by_mod = {}

    for mod in meta:
        credits, assign_pct, contact = meta[mod]
        total_notional = credits * 10

        # baseline teaching: weeks 1-6 contact, 7=0, 8-12 contact, 13-15 initially 0
        weekly = [contact]*6 + [0] + [contact]*5 + [0, 0, 0]

        # Coursework effort still to allocate into 1..12
        prep_time = max(total_notional * assign_pct - sum(weekly[:12]), 0.0)

        ca_list = ca_map.get(mod, [])
        total_pct = sum(w for (_idx, w, _dl, _rl) in ca_list) or 1.0

        for (idx, wt, dl, rl) in ca_list:
            # apply override if present
            dl = deadline_overrides.get((mod, idx), dl)

            d = dl - rl
            if d < 0 or rl < 1 or dl > 12:
                continue

            T = prep_time * (wt / total_pct)
            weights = _weights_for_span(d, study_style)

            # never put work in wk7, and keep 13-15 separate
            weekly[6] = 0
            weekly[-3:] = [0, 0, 0]

            for i, w in enumerate(weights):
                week_idx = rl - 1 + i
                if 0 <= week_idx < 12 and week_idx != 6:
                    weekly[week_idx] += T * w

        # exam distribution (weeks 13..15)
        exam_pct = 1.0 - assign_pct
        exam_effort = total_notional * exam_pct
        d_exam = 2
        exam_w = _weights_for_span(d_exam, study_style)
        for i, w in enumerate(exam_w):
            weekly[12 + i] += exam_effort * w

        weeks15_by_mod[mod] = weekly

    return weeks15_by_mod

def total_cv_percent(weeks15_by_mod: dict) -> float:
    """CV% across TOTAL for weeks 1..12 (wk7=0 by construction)."""
    # sum per week over modules
    totals = [0.0]*12
    for weekly in weeks15_by_mod.values():
        for i in range(12):
            totals[i] += weekly[i]
    mean = sum(totals)/12 if totals else 0.0
    if mean == 0:
        return 0.0
    # population std (ddof=0)
    var = sum((x - mean)**2 for x in totals) / 12
    std = var ** 0.5
    return (std / mean) * 100.0

# ---- FAST cache + scenario generator ---------------------------------
@st.cache_data(show_spinner=False)
def _weeks15_cached(study_style, meta_items, ca_map_items, overrides_items):
    """Cached call to recompute_all_weekly with hashable inputs."""
    meta = dict(meta_items)
    ca_map = {k: list(v) for k, v in ca_map_items}
    overrides = dict(overrides_items) if overrides_items else None
    return recompute_all_weekly(study_style, meta, ca_map, overrides)

def _hashables_for_cache(meta, ca_map, overrides=None):
    meta_items = tuple(sorted((k, tuple(v)) for k, v in meta.items()))
    ca_map_items = tuple(sorted((k, tuple(tuple(x) for x in v)) for k, v in ca_map.items()))
    overrides_items = tuple(sorted(overrides.items())) if overrides else None
    return meta_items, ca_map_items, overrides_items

def generate_scenarios_exact_upto_k(all_cas, Kmax, study_style, meta, ca_map, valid_fn):
    """
    Build scenarios for exactly k=1..Kmax shifts (no 0). For each chosen CA, try ±1 only.
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
# ----------------------------------------------------------------------

# ---------- end helpers ----------
    # --- Redirect anything that still points to the removed Step 2 ---
if st.session_state.step == 2:
    st.session_state.step = 3
    st.rerun()


# ─── STEP 0: Master List of Modules ───────────────────────────
if st.session_state.step == 0:
    st.title("All Modules")
    c1, c2 = st.columns([3,1])
    with c2:
     if st.button("➕ Add New Module"):
        st.session_state.selected = "__new__"
        st.session_state["_name"] = ""
        st.session_state["_credits"] = 0.0
        st.session_state["_assign_pct"] = 0.0
        st.session_state["_contact"] = 0.0
        st.session_state["_n_ca"] = 0

        # Clear previous CA values
        for k in list(st.session_state.keys()):
            if k.startswith("wt") or k.startswith("dl") or k.startswith("rel") or k.startswith("nm"):
                del st.session_state[k]


        st.session_state.step = 1
        st.stop()




    for mod in st.session_state.meta:
        with st.expander(mod):
            # pull out credits, coursework% and contact
            cr, ap, ct = st.session_state.meta[mod]
            # compute exam%
            exam_pct = 1.0 - ap
            # render all four values
            st.markdown(
                f"**Credits:** {cr}  •  "
                f"**CW %:** {ap*100:.0f}%  •  "
                f"**Exam %:** {exam_pct*100:.0f}%  •  "
                f"**Contact:** {ct}h/wk"
            )
            if st.button("✏️ Edit", key=f"edit_{mod}"):
                st.session_state.selected = mod
                st.session_state.step = 1
                st.stop()

    st.write("Click ‘Add New Module’ or ‘Edit’ to begin.")

    # if they really want to skip straight on...
    

# ─── STEP 1: Module Definitions + CAs ──────────────────────────
elif st.session_state.step == 1:
    st.title("Step 1 of 2: Module Definitions")

    # Pre-fill if editing existing
    if st.session_state.selected and st.session_state.selected != "__new__":
        sel = st.session_state.selected
        cr, ap, ct = st.session_state.meta[sel]
        st.session_state["_name"]       = sel
        st.session_state["_credits"]    = cr
        st.session_state["_assign_pct"] = ap * 100
        st.session_state["_contact"]    = ct
        st.session_state["_n_ca"]       = len(st.session_state.ca_map.get(sel, []))
        
        for idx, wt, dl, rel in st.session_state.ca_map.get(sel, []):
            st.session_state[f"wt{idx-1}"]  = wt
            st.session_state[f"dl{idx-1}"]  = dl
            st.session_state[f"rel{idx-1}"] = rel
            # NEW: name prefill
            nm_prefill = st.session_state.ca_names.get(sel, {}).get(idx, "")
            st.session_state[f"nm{idx-1}"] = nm_prefill



    with st.form("module_form"):
        name       = st.text_input("Module name", value=st.session_state.get("_name",""))
        credits    = st.number_input("Credits", min_value=0.0, step=0.5,
                                     value=st.session_state.get("_credits",0.0))
        assign_pct = st.number_input("Assignment % of total hours",
                                     min_value=0.0, max_value=100.0,
                                     value=st.session_state.get("_assign_pct",0.0))/100.0
        
                # <<< INSERTION: Coursework vs Exam % >>>
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
        assign_pct = cw_pct
        # <<< end insertion >>>


        contact    = st.number_input("Contact hrs/week",
                                     min_value=0.0, step=0.5,
                                     value=st.session_state.get("_contact",0.0))

        n_ca = st.number_input("How many CAs?", min_value=0, step=1,
                               value=st.session_state.get("_n_ca",0))
        
        ca_nms, ca_wts, ca_dls, ca_rels = [], [], [], []

        if n_ca > 0:
            st.subheader("Continuous Assessments")
            for i in range(int(n_ca)):
                # CA name
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

        # ↓↓↓ submit buttons must be INSIDE the form but OUTSIDE the loop
        b1, b2, b3 = st.columns(3)
        with b1: back = st.form_submit_button("◀ Back to list")
        with b2: save_mod = st.form_submit_button("Save / Update Module")
        with b3: alloc_cas = st.form_submit_button("Allocate CAs")

           

        # ---- handlers (outside the form) ----

    # Normalize booleans in case the form didn't define them this run
    back = bool(back) if 'back' in locals() else False
    save_mod = bool(save_mod) if 'save_mod' in locals() else False
    alloc_cas = bool(alloc_cas) if 'alloc_cas' in locals() else False

    if back:
        go_home()
        st.stop()

    if save_mod:
        if not name:
            st.error("Module name is required.")
        else:
            # Build baseline schedule once (contact weeks; wk7=0; 13–15=0 here)
            sched = [contact]*6 + [0] + [contact]*5 + [0, 0, 0]

            st.session_state.baseline[name] = sched.copy()
            st.session_state.modules[name]  = sched.copy()
            st.session_state.meta[name]     = (credits, assign_pct, contact)

            save_module_meta(name, credits, assign_pct, contact)
            save_schedule(name, sched)

            st.success(f"Module '{name}' saved.")

    if alloc_cas:
        if name not in st.session_state.modules:
            st.error("Please Save module first.")
        elif int(n_ca) == 0:
            st.warning("Set How many CAs? > 0 first.")
        else:
            # Compute allocation
            notional  = credits * assign_pct * 10
            weekly    = st.session_state.baseline[name].copy()
            prep_time = max(notional - sum(weekly[:12]), 0.0)
            total_pct = sum(ca_wts) or 1.0

            cas_list = []
            for idx, (wt, dl, rel, nm) in enumerate(zip(ca_wts, ca_dls, ca_rels, ca_nms), start=1):
                # Validate range / ordering
                if (dl - rel) < 0 or rel < 1 or dl > 12:
                    st.warning(f"Invalid release/deadline for CA#{idx} — skipped.")
                    continue

                # Persist CA (DB uses 4-tuple: idx, wt, dl, rel)
                cas_list.append((idx, wt, dl, rel))

                # Simple deadline-spike allocation (same as your previous behavior)
                weekly[dl - 1] += prep_time * (wt / total_pct)

                # Keep CA names (session)
                st.session_state.ca_names.setdefault(name, {})[idx] = nm

        # Enforce wk7 and weeks 13–15 = 0 in teaching allocation
        weekly[6]   = 0
        weekly[-3:] = [0, 0, 0]

        # Save results
        st.session_state.modules[name] = weekly
        st.session_state.ca_map[name]  = cas_list     # <— you were missing this
        save_cas(name, cas_list)                      # <— and this
        save_schedule(name, weekly)

        st.success(f"CAs allocated for '{name}'.")
    

elif st.session_state.step == 3:

    st.title("Step 2 of 2: Results")
    weeks = st.session_state.weeks
    teaching_weeks = weeks[:12]

    # ─── Study Style ─────────────────────────────────────────
    study_style = st.selectbox(
        "Study Style",
        ["Early Starter", "Steady", "Just in Time"],
        index=["Early Starter", "Steady", "Just in Time"].index(
            st.session_state.get("study_style", "Just in Time")
        )
    )
    st.session_state["study_style"] = study_style
    st.caption("This controls how effort is distributed from CA release to deadline during allocation.")

    # ─── Recompute workload (same as before) ─────────────────
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

            if study_style == "Early Starter":
                weights = [1 / (d + 1)] * (d + 1)
            elif study_style == "Steady":
                denom = (d**2 + 3*d + 2) or 1
                weights = [2 * (i + 1) / denom for i in range(d + 1)]
            else:
                weights = [0] * d + [1]

            weekly[6] = 0
            weekly[-3:] = [0, 0, 0]
            for i, w in enumerate(weights):
                week_idx = rel - 1 + i
                if 0 <= week_idx < 12:
                    weekly[week_idx] += T * w

        # exam 13–15
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

    df_main = pd.DataFrame(df_rows, index=weeks).T
    tot      = df_main[weeks].sum(axis=0)
    df_total = pd.DataFrame([tot.values], index=["TOTAL"], columns=weeks)
    df       = pd.concat([df_main, df_total], axis=0)
    df["Total"] = df[teaching_weeks].sum(axis=1)
    df["CV"]    = (df[teaching_weeks].std(ddof=0, axis=1) / df[teaching_weeks].mean(axis=1) * 100).round(1)
    df["Colour"]= df["CV"].map(colour_for_cv)

    # Styled table (hidden behind a button)
    if "show_workload" not in st.session_state:
        st.session_state.show_workload = False

    col_show, _ = st.columns([1, 7])
    if col_show.button(("Hide" if st.session_state.show_workload else "Show") + " workload table"):
        st.session_state.show_workload = not st.session_state.show_workload

    if st.session_state.show_workload:
        df_disp = df.round(1)
        styled = (
            df_disp[weeks + ["Total","CV","Colour"]]
            .style
            .applymap(lambda v: f"background-color: {colour_for_cv(v)}", subset=["CV"])
            .set_properties(color="red", subset=pd.IndexSlice[:, ["week 13","week 14","week 15"]])
        )
        st.markdown("<div style='overflow-x:auto'>" + styled.to_html() + "</div>", unsafe_allow_html=True)
    else:
        # still keep df_disp defined for Excel
        df_disp = df.round(1)

    
    # ---- compute scenarios silently for Excel (no UI) ----
    def _valid_shift(rl: int, new_dl: int) -> bool:
        return 1 <= new_dl <= 12 and new_dl >= rl

    _all_cas = []
    for _m, _cas_list in st.session_state.ca_map.items():
        for (_i, _w, _dl, _rl) in _cas_list:
            if 1 <= _dl <= 12:
                _all_cas.append((_m, _i, _rl, _dl))

    # Limit Excel scenarios to small k to keep the file useful & fast
    _excel_Kmax = min(3, len(_all_cas))  # tweak if you like
    _scenarios = generate_scenarios_exact_upto_k(
        _all_cas, _excel_Kmax, study_style,
        st.session_state.meta, st.session_state.ca_map, _valid_shift
    )


    out = (pd.DataFrame(_scenarios, columns=["no_shifts","CV","changes"])
           if _scenarios else pd.DataFrame(columns=["no_shifts","CV","changes"]))
    out = out.sort_values(["CV","no_shifts"], ascending=[True,True]).reset_index(drop=True)
    # -------------------------------------------------------

    # ─── Heatmaps entry point (unchanged) ───────────────────
    #st.markdown("### Heatmaps")
    #st.caption("See CV cross-heatmaps for every pair of modules on one page.")
    if st.button("Show all module-pair heatmaps"):
        st.session_state.heatmap_modules = "__ALL__"
        st.session_state.step = "HEATMAP"
        st.rerun()

    # ─── INLINE CALENDAR (moved from Step 4) ─────────────────
    st.markdown("### Calendar")
    # compute current CV
    base_weeks15 = recompute_all_weekly(study_style, st.session_state.meta, st.session_state.ca_map)
    base_cv = total_cv_percent(base_weeks15)
    st.markdown(f"<p class='caption-lg'>Pain score (CV weeks 1–12): <b>{base_cv:.1f}%</b></p>", unsafe_allow_html=True)

    # calendar data/formatting helpers reused from Step 4
    cal_df = build_calendar_df(st.session_state.weeks)
    cal_df_display = cal_df.reset_index()
    week_cols = [c for c in cal_df_display.columns if c.startswith("week")]

    n_weeks = len(week_cols)
    module_pct = 16
    week_pct   = (100 - module_pct) / n_weeks

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

    preview = st.session_state.get("preview_overrides", {})
    weeks15_by_mod = recompute_all_weekly(study_style, st.session_state.meta, st.session_state.ca_map,
                                          preview if preview else None)

    def _cv_percent_1to12(weekly):
        vals = weekly[:12]
        m = sum(vals)/12 if vals else 0.0
        if m == 0: return 0.0
        var = sum((x - m)**2 for x in vals) / 12
        return (var ** 0.5) / m * 100.0

    band_hex = {"green":"#2ecc71","yellow":"#f1c40f","orange":"#e67e22","red":"#e74c3c","black":"#000000"}
    row_band = {m: colour_for_cv(_cv_percent_1to12(w)) for m,w in weeks15_by_mod.items()}

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

    def _stripe_module_col(col: pd.Series) -> list[str]:
        styles = []
        for i, _ in enumerate(col):
            module = cal_df_display.iloc[i]["Module"]
            stripe = band_hex.get(row_band.get(module, "green"), "#2ecc71")
            styles.append(f"border-left: 8px solid {stripe};")
        return styles

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

    # --- Number-of-shifts selector: typeable + Show button ---
    st.subheader("Recommendations")

    if "N" not in st.session_state:
        st.session_state.N = 2          # persistent chosen number (1..10)

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



    # --- Undo + Recommendations UI (same behavior as Step 4) ---
    if st.session_state.get("undo_payload"):
        with st.container(border=True):
            st.write("**Undo last shift**")
            if st.session_state.get("undo_caption"):
                st.caption(st.session_state.undo_caption)
            if st.button("Undo"):
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

    # Build scenarios for the visible calendar page (like Step 4)
    all_cas = []
    for mod, cas_list in st.session_state.ca_map.items():
        for (idx, wt, dl, rl) in cas_list:
            if 1 <= dl <= 12:
                all_cas.append((mod, idx, rl, dl))

    # Generate only up to the user’s chosen N; we’ll filter to exactly N below.
    scenarios = generate_scenarios_exact_upto_k(
        all_cas, st.session_state.N, study_style,
        st.session_state.meta, st.session_state.ca_map, _valid_shift
    )


    st.subheader("Recommendations")

    

    if scenarios:
        # Build dataframe and filter by the chosen number of shifts (N)
        # Build dataframe and filter by the chosen number of shifts (EXACTLY N)
        df_scen = pd.DataFrame(scenarios, columns=["no_shifts","CV","changes"])
        filtered = df_scen.query("no_shifts == @st.session_state.N")

        out_visible = (
            filtered
            .sort_values(["CV","no_shifts"], ascending=[True, True])
            .reset_index(drop=True)
        )


        topN = out_visible.head(5).copy()



        for i, row in topN.iterrows():
            with st.container(border=True):
                st.write(f"**Option {i+1}** — CV **{row['CV']:.1f}%**")
                st.caption(row["changes"] if row["changes"] != "none" else "No changes")
                c1, c2, _ = st.columns([1,1,6])

                # add this line:
                uniq = f"{i}_{abs(hash(row['changes'])) % 10_000_000}"

                if c1.button("Visualize", key=f"viz_cal_{uniq}"):
                    st.session_state["preview_overrides"] = {
                        (p.split(' CA#')[0].strip(), int(p.split(' CA#')[1].split('@')[0])):
                        int(p.split('→')[1].strip())
                        for p in [x.strip() for x in row["changes"].split(';')] if p and "→" in p
                    }
                    st.rerun()

                if c2.button("Shift now", key=f"apply_cal_{uniq}"):
                    overrides = {
                        (p.split(' CA#')[0].strip(), int(p.split(' CA#')[1].split('@')[0])):
                        int(p.split('→')[1].strip())
                        for p in [x.strip() for x in row["changes"].split(';')] if p and "→" in p
                    }
                    if not overrides:
                        st.info("No changes to apply.")
                    else:
       
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
                        for (m, idx), new_dl in overrides.items():
                            old_list = st.session_state.ca_map.get(m, [])
                            new_list = []
                            for (j, wt, dl, rl) in old_list:
                                if j == idx:
                                    dl = new_dl
                                new_list.append((j, wt, dl, rl))
                            st.session_state.ca_map[m] = new_list
                            save_cas(m, new_list)
                        st.session_state.pop("preview_overrides", None)
                        st.success("Shifts applied. Calendar shows old location shaded with an arrow old→new.")
                        st.rerun()
    else:
        st.info("No candidate scenarios available with ±1 week shifts.")

    if st.session_state.get("preview_overrides"):
        if st.button("Reset preview"):
            st.session_state.pop("preview_overrides", None)
            st.rerun()

    # ─── Download Excel ───────────────────────────────────────
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

   

    # (optional) Workload chart (always visible here)
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



# ─── HEATMAP PAGE (diverted from Step 3) ─────────────────────────────────────────
elif st.session_state.step == "HEATMAP":
    flag = st.session_state.get("heatmap_modules")
    style = st.session_state.get("study_style", "Just in Time")

    # ------- ALL-PAIRS PAGE -------
    if flag == "__ALL__":
 
    

        # helper to check if module has any CA deadlines in weeks 1–12
        def _has_teaching_ca(mod):
            return any(1 <= dl <= 12 for (_i, _w, dl, _r) in st.session_state.ca_map.get(mod, []))

        mods_with_cas = [m for m in st.session_state.meta.keys() if _has_teaching_ca(m)]
        pairs = list(itertools.combinations(mods_with_cas, 2))

        st.title("All module-pair heatmaps")

        # 👉 NEW: wrap your Step 4 body inside this function
        def render_pair(A_mod, B_mod, highlight_coords=None, compute_only=False):
           # don’t draw headings during compute-only pass
            if not compute_only:
             st.markdown(f"### {A_mod} vs {B_mod} — CV cross heatmap")


            A_cas = [(idx, wt, dl, rl) for (idx, wt, dl, rl) in st.session_state.ca_map.get(A_mod, []) if 1 <= dl <= 12]
            B_cas = [(idx, wt, dl, rl) for (idx, wt, dl, rl) in st.session_state.ca_map.get(B_mod, []) if 1 <= dl <= 12]

             # NEW: define shading sets here
            A_weeks_with_ca = {dl for (_idx, _wt, dl, _rl) in A_cas}
            B_weeks_with_ca = {dl for (_idx, _wt, dl, _rl) in B_cas}


            if not A_cas or not B_cas:
               if compute_only:
             # return empty df + None so caller can skip it quietly
                 return pd.DataFrame(), None
               st.info("One or both selected modules have no CAs with deadlines in weeks 1–12.")
               return


            # --- helpers
            def cv_band_and_color(cv: float):
                if cv < 40:           return "<40",      "#2ecc71"
                elif cv < 50:         return "40–49.9",  "#f1c40f"
                elif cv < 65:         return "50–64.9",  "#e67e22"
                elif cv < 80:         return "65–79.9",  "#e74c3c"
                else:                 return "≥80",      "#b71c1c"


            def can_shift(mod: str, w: int, dir_: int) -> bool:
                new_w = w + dir_
                if not (1 <= new_w <= 12):
                    return False
                for (_idx, _wt, dl, rl) in st.session_state.ca_map.get(mod, []):
                    if dl == w and new_w < rl:
                        return False
                return True

            # collisions (same-week deadlines)
            A_dead = {dl for (_i, _w, dl, _r) in A_cas} if A_cas else set()
            B_dead = {dl for (_i, _w, dl, _r) in B_cas} if B_cas else set()
            collisions = sorted(A_dead & B_dead)

            # base CV
            base_weeks = recompute_all_weekly(style, st.session_state.meta, st.session_state.ca_map)
            center_cv = total_cv_percent(base_weeks)

            # grid data store
            grid = {(a, b): {"cv": None, "count": 0, "label": "", "band": ""} for a in range(1, 13) for b in range(1, 13)}

            def set_invalid(a, b):
                cell = grid[(a, b)]
                if cell["label"] == "":
                    cell["label"] = "–"

            def put_cv(a, b, cv_val: float):
                band, _ = cv_band_and_color(cv_val)
                cell = grid[(a, b)]
                if cell["cv"] is None or cv_val < cell["cv"] - 1e-9:
                    cell["cv"] = cv_val
                    cell["count"] = 1          # keep the tie count internally if you want tooltips later
                    cell["band"] = band
                    cell["label"] = f"{cv_val:.1f}"   # <-- no ×n in the label
                elif abs(cv_val - cell["cv"]) <= 1e-9:
                    cell["count"] += 1
                    cell["label"] = f"{cell['cv']:.1f}"   # <-- still no ×n

            # fill crosses for each collision
            for w in collisions:
                put_cv(w, w, center_cv)  # center
                # shift A
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
                # shift B
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
                        # shift BOTH A and B together (diagonals): (w±1, w±1)
                for dA in (-1, 1):
                    for dB in (-1, 1):
                        a_w = w + dA
                        b_w = w + dB
                        if can_shift(A_mod, w, dA) and can_shift(B_mod, w, dB):
                            overA = {
                                (A_mod, idx): a_w
                                for (idx, wt, dl, rl) in st.session_state.ca_map.get(A_mod, [])
                                if dl == w
                            }
                            overB = {
                                (B_mod, idx): b_w
                                for (idx, wt, dl, rl) in st.session_state.ca_map.get(B_mod, [])
                                if dl == w
                            }
                            overrides = {}
                            overrides.update(overA)
                            overrides.update(overB)

                            weeks_shift = recompute_all_weekly(
                                style, st.session_state.meta, st.session_state.ca_map, overrides
                            )
                            put_cv(a_w, b_w, total_cv_percent(weeks_shift))
                        else:
                            # show a dash if the cell is inside the 12×12 lattice but invalid
                            if 1 <= a_w <= 12 and 1 <= b_w <= 12:
                                set_invalid(a_w, b_w)
                    # fill crosses for each collision
            for w in collisions:
                # center
                put_cv(w, w, center_cv)

                # ±1 orthogonals: A moves, B stays
                for dir_ in (-1, 1):
                    a_w = w + dir_
                    if can_shift(A_mod, w, dir_):
                        overrides = {(A_mod, idx): a_w
                                for (idx, wt, dl, rl) in st.session_state.ca_map.get(A_mod, [])
                                if dl == w}
                        weeks_shift = recompute_all_weekly(style, st.session_state.meta, st.session_state.ca_map, overrides)
                        put_cv(a_w, w, total_cv_percent(weeks_shift))
                    elif 1 <= a_w <= 12:
                        set_invalid(a_w, w)

                # ±1 orthogonals: B moves, A stays
                for dir_ in (-1, 1):
                    b_w = w + dir_
                    if can_shift(B_mod, w, dir_):
                        overrides = {(B_mod, idx): b_w
                                for (idx, wt, dl, rl) in st.session_state.ca_map.get(B_mod, [])
                                if dl == w}
                        weeks_shift = recompute_all_weekly(style, st.session_state.meta, st.session_state.ca_map, overrides)
                        put_cv(w, b_w, total_cv_percent(weeks_shift))
                    elif 1 <= b_w <= 12:
                        set_invalid(w, b_w)

                # ±1 diagonals: A and B both move
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
                            weeks_shift = recompute_all_weekly(style, st.session_state.meta, st.session_state.ca_map, overrides)
                            put_cv(a_w, b_w, total_cv_percent(weeks_shift))
                        elif 1 <= a_w <= 12 and 1 <= b_w <= 12:
                            set_invalid(a_w, b_w)

                # ── NEW: ring-2 orthogonals (your yellow blocks): (w±2, w) and (w, w±2)
                for step in (-2, 2):
                    # A moves ±2, B stays
                    a_w = w + step
                    if can_shift(A_mod, w, step):
                        overA = {(A_mod, idx): a_w
                                for (idx, wt, dl, rl) in st.session_state.ca_map.get(A_mod, [])
                                if dl == w}
                        weeks_shift = recompute_all_weekly(style, st.session_state.meta, st.session_state.ca_map, overA)
                        put_cv(a_w, w, total_cv_percent(weeks_shift))
                    elif 1 <= a_w <= 12:
                        set_invalid(a_w, w)

                    # B moves ±2, A stays
                    b_w = w + step
                    if can_shift(B_mod, w, step):
                        overB = {(B_mod, idx): b_w
                                for (idx, wt, dl, rl) in st.session_state.ca_map.get(B_mod, [])
                                if dl == w}
                        weeks_shift = recompute_all_weekly(style, st.session_state.meta, st.session_state.ca_map, overB)
                        put_cv(w, b_w, total_cv_percent(weeks_shift))
                    elif 1 <= b_w <= 12:
                        set_invalid(w, b_w)


            # ---------- RENDER (full lattice + banded fill + diagonal + numbers) ----------
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
                rows.append({"A_week": a, "B_week": b, "Label": label, "Band": band,
                            "Color": color, "IsDash": (label == "–"), "HasBand": band != "", 
                            "RowHasA": (a in A_weeks_with_ca),
                            "ColHasB": (b in B_weeks_with_ca),})

            grid_df = pd.DataFrame(rows)
            # Convert Label (strings like "54.3" or "–" / "") into numeric CV values
            grid_df["CVnum"] = pd.to_numeric(grid_df["Label"], errors="coerce")

            # Mark collisions: cells on the diagonal that correspond to real same‑week deadlines
            grid_df["IsCollision"] = grid_df.apply(
                lambda r: (int(r["A_week"]) == int(r["B_week"])) and (int(r["A_week"]) in collisions),
                axis=1
            )


                        # Pair minimum CV (if present)
            pair_min_row = grid_df.dropna(subset=["CVnum"]).nsmallest(1, "CVnum")
            pair_min_cv = float(pair_min_row.iloc[0]["CVnum"]) if not pair_min_row.empty else None

            # If compute_only, return the data needed by the caller and skip rendering
            if compute_only:
                return grid_df, pair_min_cv

            # Mark all coordinates that should be highlighted (may be many)
            if highlight_coords:
                highlight_set = set(highlight_coords)
                grid_df["IsGlobalBest"] = grid_df.apply(
                    lambda r: (int(r["A_week"]), int(r["B_week"])) in highlight_set, axis=1
                )
            else:
                grid_df["IsGlobalBest"] = False


            band_domain = ["<40", "40–49.9", "50–64.9", "65–79.9", "≥80"]
            band_colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#b71c1c"]

            # sizing for spacious layout
            cell_size = 60   # try 60–80 for bigger cells
            chart_width = cell_size * 12
            chart_height = cell_size * 12

            # reverse the Y axis so B_week shows 12 at the top
            Y_DOMAIN_DESC = list(range(12, 0, -1))


            # NEW: light shading so weeks with any assignment are visible even without collisions
            row_shade = (
                alt.Chart(grid_df)
                .mark_rect(opacity=0.28, fill="#42a5f5", stroke=None)  # darker blue, more visible
                .encode(
                    x=alt.X("A_week:O", scale=alt.Scale(domain=list(range(1,13))), axis=alt.Axis(labels=False, ticks=False)),
                    y=alt.Y("B_week:O", scale=alt.Scale(domain=Y_DOMAIN_DESC),        axis=alt.Axis(labels=False, ticks=False)),
                )
                .transform_filter("datum.RowHasA == true")
                .properties(width=chart_width, height=chart_height)
            )

            col_shade = (
                alt.Chart(grid_df)
                .mark_rect(opacity=0.24, fill="#ffb74d", stroke=None)  # darker amber, more visible
                .encode(
                    x=alt.X("A_week:O", scale=alt.Scale(domain=list(range(1,13))), axis=alt.Axis(labels=False, ticks=False)),
                    y=alt.Y("B_week:O", scale=alt.Scale(domain=Y_DOMAIN_DESC),        axis=alt.Axis(labels=False, ticks=False)),
                )
                .transform_filter("datum.ColHasB == true")
                .properties(width=chart_width, height=chart_height)
            )


            # 1) lattice outlines (bottom & left axes shown here)
            lattice = (
            alt.Chart(grid_df)
            .mark_rect(fillOpacity=0, stroke="#9fb3c0", strokeWidth=1)
            .encode(
                x=alt.X(
                    "A_week:O",
                    title=f"{A_mod} — deadline week (1–12)",
                    scale=alt.Scale(domain=list(range(1,13))),
                    axis=alt.Axis(orient='bottom', labelAngle=0, labelFontSize=12, titleFontSize=13, ticks=True),
                ),  # ← comma here
                y=alt.Y(
                    "B_week:O",
                    title=f"{B_mod} — deadline week (1–12)",
                    scale=alt.Scale(domain=Y_DOMAIN_DESC),
                    axis=alt.Axis(orient='left', labelAngle=0, labelFontSize=12, titleFontSize=13, ticks=True),
                ),
            )
            .properties(width=chart_width, height=chart_height)
        )


            # 2) banded fill
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
                                legend=alt.Legend(title="CV band (TOTAL, weeks 1–12)"),
                                scale=alt.Scale(domain=band_domain, range=band_colors)),
            )
            .transform_filter("datum.HasBand == true")
            .properties(width=chart_width, height=chart_height)
        )


            # 4) numeric labels
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
            
            # Blue border for collision cells (thicker than black)
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


            # Visible box around the winning cell (drawn only if IsGlobalBest == True)
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
                                        alt.value("#000000"),  # black box
                                        alt.value(None))
                )
                .transform_filter("datum.IsGlobalBest == true")
                .properties(width=chart_width, height=chart_height)
            )


            # EXTRA: duplicate axes to show top X and right Y (no visual marks)
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

            # Legend (key) for border meanings
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

            # now loop all pairs
        if not pairs:
            st.info("No module pairs with CAs in weeks 1–12 were found.")
        else:
            # ---------- PASS 1: compute global minimum (no rendering) ----------
            pair_min_cv_map = {}  # (A_mod, B_mod) -> min CV for that pair (float or None)
            for A_mod, B_mod in pairs:
                grid_df, pair_min_cv = render_pair(A_mod, B_mod, compute_only=True)
                pair_min_cv_map[(A_mod, B_mod)] = pair_min_cv

            valid_mins = [v for v in pair_min_cv_map.values() if v is not None]
            global_min_cv = min(valid_mins) if valid_mins else None

            if global_min_cv is None:
                st.warning("No numeric CV values found to highlight.")
            else:
                st.success(f"🌟 Global best CV is **{global_min_cv:.1f}%** (highlighted in all heatmaps below).")

            # ---------- PASS 2: render all pairs, highlighting every global-min cell ----------
            tol = 1e-9  # tolerance for float comparisons
            for A_mod, B_mod in pairs:
                highlight_coords = None
                if global_min_cv is not None:
                    # recompute this pair's grid to find all cells equal to the global min
                    grid_df, _ = render_pair(A_mod, B_mod, compute_only=True)
                    hits = grid_df.loc[
                        grid_df["CVnum"].notna() & (abs(grid_df["CVnum"] - global_min_cv) <= tol),
                        ["A_week", "B_week"]
                    ]
                    if not hits.empty:
                        highlight_coords = [(int(a), int(b)) for a, b in hits.to_numpy()]

                # Now actually render (with or without highlights)
                render_pair(A_mod, B_mod, highlight_coords=highlight_coords)
                st.divider()

        if st.button("◀ Previous"):
            st.session_state.step = 3
            st.rerun()

        st.stop()

    # --- fallback: your original single-pair Step 4 code ---
    pair = st.session_state.get("heatmap_modules")
    if not pair or len(pair) != 2:
        st.error("No module pair selected.")
    else:
        A_mod, B_mod = pair

        st.markdown(f"### {A_mod} vs {B_mod} — CV cross heatmap")
        A_cas = [(idx, wt, dl, rl) for (idx, wt, dl, rl) in st.session_state.ca_map.get(A_mod, []) if 1 <= dl <= 12]
        B_cas = [(idx, wt, dl, rl) for (idx, wt, dl, rl) in st.session_state.ca_map.get(B_mod, []) if 1 <= dl <= 12]

        # NEW: weeks that have at least one CA (used for row/column shading)
        A_weeks_with_ca = {dl for (_idx, _wt, dl, _rl) in A_cas}
        B_weeks_with_ca = {dl for (_idx, _wt, dl, _rl) in B_cas}

        # … the same Step 4 logic as before …
       # st.altair_chart(chart, use_container_width=False)

        if not collisions:
            st.caption("No colliding deadlines for the selected modules (weeks 1–12).")

    if st.button("◀ Previous"):
        st.session_state.step = 3
        st.rerun()

    