# db.py  — pg8000 version (Windows-friendly)
import os
import json
import ssl
from urllib.parse import urlparse, parse_qs
from pg8000 import dbapi as pg


def _conn():
    """
    Create a pg8000 DB-API connection from DATABASE_URL.
    Works with Render-style URLs (…?sslmode=require).
    """
    url = os.environ["DATABASE_URL"]
    # Accept both postgres:// and postgresql://
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://") :]

    p = urlparse(url)
    user = p.username
    password = p.password
    host = p.hostname
    port = p.port or 5432
    database = (p.path or "/").lstrip("/")
    qs = parse_qs(p.query or "")

    # SSL if requested (Render uses sslmode=require)
    ctx = None
    if qs.get("sslmode", [""])[0] in ("require", "verify-ca", "verify-full"):
        ctx = ssl.create_default_context()

    return pg.connect(
        user=user,
        password=password,
        host=host,
        port=port,
        database=database,
        ssl_context=ctx,
    )


def _fetchall_dict(cur):
    """Return rows as list[dict] so the rest of the app can do r['col']."""
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def ensure_schema():
    with _conn() as conn:
        cur = conn.cursor()
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
    """Return (modules_meta_dict, cas_map_dict, schedules_dict)."""
    with _conn() as conn:
        cur = conn.cursor()

        # modules_meta: name -> (credits, assign_pct, contact)
        cur.execute("SELECT name, credits, assign_pct, contact FROM modules_meta;")
        meta_rows = _fetchall_dict(cur)
        modules_meta = {
            r["name"]: (float(r["credits"]), float(r["assign_pct"]), float(r["contact"]))
            for r in meta_rows
        }

        # cas_map: module -> [(idx, wt, dl, rl), ...]
        cur.execute("SELECT module, idx, wt, dl, rl FROM cas_map ORDER BY module, idx;")
        cas_rows = _fetchall_dict(cur)
        cas_map = {}
        for r in cas_rows:
            cas_map.setdefault(r["module"], []).append(
                (int(r["idx"]), float(r["wt"]), int(r["dl"]), int(r["rl"]))
            )

        # schedules: module -> [list of 15 numbers]
        cur.execute("SELECT module, weekly FROM schedules;")
        sch_rows = _fetchall_dict(cur)
        schedules = {r["module"]: list(r["weekly"]) for r in sch_rows}

        return modules_meta, cas_map, schedules


def save_module_meta(name, credits, assign_pct, contact):
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO modules_meta (name, credits, assign_pct, contact)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (name) DO UPDATE SET
                credits=EXCLUDED.credits,
                assign_pct=EXCLUDED.assign_pct,
                contact=EXCLUDED.contact;
            """,
            (name, credits, assign_pct, contact),
        )
        conn.commit()


def save_cas(name, cas_list):
    """cas_list: list of (idx, wt, dl, rl). Replace all for module."""
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM cas_map WHERE module=%s;", (name,))
        if cas_list:
            cur.executemany(
                """
                INSERT INTO cas_map (module, idx, wt, dl, rl)
                VALUES (%s, %s, %s, %s, %s);
                """,
                [(name, int(i), float(w), int(dl), int(rl)) for (i, w, dl, rl) in cas_list],
            )
        conn.commit()


def save_schedule(name, weekly):
    """weekly: list of 15 numbers."""
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO schedules (module, weekly)
            VALUES (%s, %s::jsonb)
            ON CONFLICT (module) DO UPDATE SET weekly=EXCLUDED.weekly;
            """,
            (name, json.dumps(list(weekly))),
        )
        conn.commit()
