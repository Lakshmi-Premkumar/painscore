# db.py
import os, json
import psycopg
from psycopg.rows import dict_row

DATABASE_URL = os.environ["DATABASE_URL"]  # Render will inject this
# Tip: if you want to run locally without Postgres, you can set a local DATABASE_URL env var.

def _conn():
    # psycopg 3 connects with SSL if present in URL (Render uses sslmode=require)
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)

def ensure_schema():
    with _conn() as conn, conn.cursor() as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS modules_meta (
            name TEXT PRIMARY KEY,
            credits REAL NOT NULL,
            assign_pct REAL NOT NULL,
            contact REAL NOT NULL
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS cas_map (
            module TEXT NOT NULL,
            idx INTEGER NOT NULL,
            wt REAL NOT NULL,
            dl INTEGER NOT NULL,
            rl INTEGER NOT NULL,
            PRIMARY KEY (module, idx),
            FOREIGN KEY (module) REFERENCES modules_meta(name) ON DELETE CASCADE
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS schedules (
            module TEXT PRIMARY KEY,
            weekly JSONB NOT NULL,
            FOREIGN KEY (module) REFERENCES modules_meta(name) ON DELETE CASCADE
        );
        """)
        conn.commit()

def load_all():
    """Return (modules_meta_dict, cas_map_dict, schedules_dict) matching your app."""
    with _conn() as conn, conn.cursor() as cur:
        # modules_meta: name -> (credits, assign_pct, contact)
        cur.execute("SELECT name, credits, assign_pct, contact FROM modules_meta;")
        meta_rows = cur.fetchall()
        modules_meta = {r["name"]: (float(r["credits"]), float(r["assign_pct"]), float(r["contact"]))
                        for r in meta_rows}

        # cas_map: module -> [(idx, wt, dl, rl), ...]
        cur.execute("SELECT module, idx, wt, dl, rl FROM cas_map ORDER BY module, idx;")
        cas_rows = cur.fetchall()
        cas_map = {}
        for r in cas_rows:
            cas_map.setdefault(r["module"], []).append(
                (int(r["idx"]), float(r["wt"]), int(r["dl"]), int(r["rl"]))
            )

        # schedules: module -> [list of 15 numbers]
        cur.execute("SELECT module, weekly FROM schedules;")
        sch_rows = cur.fetchall()
        schedules = {r["module"]: list(r["weekly"]) for r in sch_rows}

        return modules_meta, cas_map, schedules

def save_module_meta(name, credits, assign_pct, contact):
    with _conn() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO modules_meta (name, credits, assign_pct, contact)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (name) DO UPDATE SET
                credits=EXCLUDED.credits,
                assign_pct=EXCLUDED.assign_pct,
                contact=EXCLUDED.contact;
        """, (name, credits, assign_pct, contact))
        conn.commit()

def save_cas(name, cas_list):
    """cas_list: list of (idx, wt, dl, rl). Replace all for module."""
    with _conn() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM cas_map WHERE module=%s;", (name,))
        if cas_list:
            cur.executemany("""
                INSERT INTO cas_map (module, idx, wt, dl, rl)
                VALUES (%s, %s, %s, %s, %s);
            """, [(name, int(i), float(w), int(dl), int(rl)) for (i, w, dl, rl) in cas_list])
        conn.commit()

def save_schedule(name, weekly):
    """weekly: list of 15 numbers."""
    with _conn() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO schedules (module, weekly)
            VALUES (%s, %s::jsonb)
            ON CONFLICT (module) DO UPDATE SET weekly=EXCLUDED.weekly;
        """, (name, json.dumps(list(weekly))))
        conn.commit()
