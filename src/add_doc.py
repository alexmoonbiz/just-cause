"""Save a link-only page as a SUPPLEMENTARY research text.

Organizer rule: texts you save yourself may be used for research if the site's terms allow it, but
they do NOT count toward the citation metric (only supplied, verifiable corpus text does). The
pipeline therefore keeps them out of the submission unless you opt in:

    python src/extract.py --include-supplementary
    python src/verify.py --with-supplementary      # without this flag they go to out/rules_supplementary.json


    python src/add_doc.py D060 --url https://... --file saved_page.txt
    python src/add_doc.py D032 --url https://ecode360.com/... --file hoboken.txt

Writes corpus/text/<doc_id>.txt with the SOURCE/RETRIEVED header the pipeline expects.
"""
import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("doc_id")
p.add_argument("--url", required=True)
p.add_argument("--file", type=Path, required=True)
p.add_argument("--jurisdiction", default="")
a = p.parse_args()

body = a.file.read_text(encoding="utf-8", errors="replace").strip()
now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
out = ROOT / "corpus" / "text" / f"{a.doc_id}.txt"
out.write_text(f"SOURCE: {a.url}\nRETRIEVED: {now}\n\n{body}\n", encoding="utf-8")
print(f"wrote {out} ({len(body):,} chars)" + (f"; jurisdiction hint: {a.jurisdiction}" if a.jurisdiction else ""))
if a.jurisdiction:
    print("tip: put the jurisdiction in the first line of the file too if the manifest has no row for this id")
