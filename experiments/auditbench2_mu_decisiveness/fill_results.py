#!/usr/bin/env python3
"""Fill the HTML-comment placeholders in RESULTS.md from collect_results.sh output.
usage: fill_results.py [--results RESULTS.md] [--tables /workspace/auditbench-2/results] [--pod-hours H] [--interpretation FILE]
Placeholders: <!-- GPU_TABLE --> (table.md), <!-- EXACT_TABLE --> (exact_table.md), <!-- POD_HOURS -->, <!-- GPU_INTERPRETATION --> (text file).
Idempotent: placeholders already replaced are left alone; unknown ones are reported."""
import argparse, pathlib, re
ap = argparse.ArgumentParser(); ap.add_argument("--results", default=str(pathlib.Path(__file__).with_name("RESULTS.md")))
ap.add_argument("--tables", default="/workspace/auditbench-2/results"); ap.add_argument("--pod-hours"); ap.add_argument("--interpretation")
a = ap.parse_args(); r = pathlib.Path(a.results); s = r.read_text(); t = pathlib.Path(a.tables)
subs = {"GPU_TABLE": (t / "table.md"), "EXACT_TABLE": (t / "exact_table.md")}
for k, f in subs.items():
    if f"<!-- {k} -->" in s and f.exists(): s = s.replace(f"<!-- {k} -->", f.read_text().rstrip("\n")); print("filled", k, "from", f)
if a.pod_hours and "<!-- POD_HOURS -->" in s: s = s.replace("<!-- POD_HOURS -->", a.pod_hours); print("filled POD_HOURS")
if a.interpretation and "<!-- GPU_INTERPRETATION -->" in s:
    s = s.replace("<!-- GPU_INTERPRETATION -->", pathlib.Path(a.interpretation).read_text().rstrip("\n")); print("filled GPU_INTERPRETATION")
left = re.findall(r"<!-- (\w+) -->", s); r.write_text(s); print("remaining placeholders:", left or "none")
tok = re.findall(r"«[^»]*»", s); print("UNFILLED «» tokens:", tok) if tok else print("no «» tokens left")
