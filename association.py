"""
Association rules between programs (and messages) found in the chunks.

    python association.py chunks_titled.jsonl                 # baskets = records (default)
    python association.py chunks_titled.jsonl --unit document
    python association.py chunks_titled.jsonl --unit chunk --messages

A "basket" is a group of text; its "items" are the program codes it mentions
(and, with --messages, the KND-/ORA- codes). Two items that appear in the same
basket much more often than chance become a rule  X -> Y.

    --unit chunk     one chunk                    (strict: must be mentioned close together)
    --unit topic     one numbered topic, otherwise one record
    --unit record    one pedido / e-mail / documentation entry   (default)
    --unit document  one whole document           (loose: many weak pairs)

Measures (per rule X -> Y)
    together    baskets that contain both X and Y
    documents   distinct documents behind those baskets (protects against one
                document repeating the same pair many times)
    confidence  together / baskets with X        "when X appears, how often Y also does"
    lift        confidence / (share of baskets with Y)   >1 = more than chance
    conviction  (1 - share Y) / (1 - confidence)  how much X "implies" Y

Outputs in folder "assoc_out":
    rules.csv         all rules that pass the thresholds, strongest first
    items.csv         each item: baskets, documents, number of rules
    association.html  type one or more programs (as a pedido would change them) and
                      see what the rules recommend, with the chunks that support each rule
Only the Python standard library is used.
"""
import argparse, collections, csv, itertools, json, math, os, re

ap = argparse.ArgumentParser()
ap.add_argument("chunks", nargs="?", default="chunks_titled.jsonl")
ap.add_argument("--unit", choices=["chunk", "topic", "record", "document"], default="record")
ap.add_argument("--messages", action="store_true", help="also use KND-/ORA- message codes as items")
ap.add_argument("--terms", help="business-term dictionary (e.g. terms.txt); terms become items named #term")
ap.add_argument("--term-term", action="store_true",
                help="also keep term->term rules (off: they mostly repeat that 'nota fiscal de saída' is a 'nota fiscal')")
ap.add_argument("--term-confidence", type=float, default=0.03,
                help="min confidence for term->program rules: a term like 'estoque' appears in thousands of "
                     "baskets, so only a small share of them can mention any single program")
ap.add_argument("--term-lift", type=float, default=3.0, help="min lift for term->program rules")
ap.add_argument("--min-together", type=int, default=3)
ap.add_argument("--min-docs", type=int, default=3)
ap.add_argument("--min-confidence", type=float, default=0.2)
ap.add_argument("--min-lift", type=float, default=2.0)
ap.add_argument("--max-items", type=int, default=25,
                help="ignore baskets with more items than this (lists/indexes that mention everything)")
ap.add_argument("--out", default="assoc_out")
ap.add_argument("--menu", default="DADOS_MENU.csv",
                help="ERP menu export (PROGRAMA, DESCRICAO, TIPO): official program names, whether each "
                     "one is a program or a report, and the menu variants")
ap.add_argument("--descriptions", default="program_descriptions.csv",
                help="optional CSV 'program;description' written by your team; it is shown first in each "
                     "program's description (a template with every program is written to <out>)")
ap.add_argument("--db", default=None, help="SQLite database to write (default: <out>/associations.db)")
ap.add_argument("--version", default=None,
                help="label of this run inside the database (default: date and time). "
                     "Running again with the same label replaces that run")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

MSG = re.compile(r"\b(?:KND|ORA)-\s?\d+")

import unicodedata
def fold(s):
    return "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn")

def load_terms(path):
    """terms.txt -> one regex that finds every variant, longest variants first."""
    canon = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line: continue
        name, variants = line.split(":", 1)
        for v in [name] + variants.split(","):
            v = fold(v.strip())
            if v: canon[v] = name.strip()
    parts = []
    for v in sorted(canon, key=len, reverse=True):
        pat = re.escape(v.rstrip("*")).replace(r"\ ", r"\s+")
        parts.append(pat + (r"\w*" if v.endswith("*") else ""))
    rx = re.compile(r"(?<!\w)(?:" + "|".join(parts) + r")(?!\w)")
    lookup = [(re.compile(r"(?:" + (re.escape(v.rstrip("*")).replace(r"\ ", r"\s+")) +
                          (r"\w*" if v.endswith("*") else "") + r")$"), n)
              for v, n in sorted(canon.items(), key=lambda x: -len(x[0]))]
    def find(text):
        found = set()
        clean = re.sub(r"\[(?:Imagem|Código):[^\]]*\]", " ", fold(text))
        for m in rx.finditer(clean):
            w = m.group(0)
            for lr, n in lookup:
                if lr.match(w):
                    found.add("#" + n); break
        return found
    return find

find_terms = load_terms(args.terms) if args.terms else None
MOD = re.compile(r"^[A-Z]+")

# ------------------------------------------------------------------ baskets
rows = [json.loads(l) for l in open(args.chunks, encoding="utf-8")]

def basket_key(r):
    if args.unit == "chunk": return r["chunk_id"]
    if args.unit == "document": return str(r["doc_id"])
    if args.unit == "topic" and r.get("topic_number"): return r["topic_id"]
    return f'{r["doc_id"]}-r{r["record"]}'

baskets = collections.defaultdict(set)
basket_doc, basket_chunks = {}, collections.defaultdict(list)
seen_content = set()
duplicates = 0
for r in rows:
    k = basket_key(r)
    items = set(r["mentioned_programs"].split()) | set(r["main_programs"].split())
    if find_terms:
        items |= find_terms(r["text"])
    if args.messages:
        items |= {re.sub(r"\s", "", m) for m in r.get("messages", [])}
        items |= {re.sub(r"\s", "", m) for m in MSG.findall(r["text"])}
    # the export has copies of the same document (and of the same pages); a copied
    # chunk must not count twice, or its pairs look much stronger than they are
    if len(items) >= 2:
        fp = (frozenset(items), re.sub(r"\W+", "", r["text"].lower())[:150])
        if fp in seen_content:
            duplicates += 1
            continue
        seen_content.add(fp)
    baskets[k] |= items
    basket_doc[k] = r["doc_id"]
    basket_chunks[k].append((r["chunk_id"], items))

too_big = [k for k, v in baskets.items()
           if sum(1 for i in v if not i.startswith("#")) > args.max_items]   # terms don't count
for k in too_big: del baskets[k]
N = len(baskets)
docs_total = len({basket_doc[k] for k in baskets})

# ------------------------------------------------------------------ counting
item_n = collections.Counter()
item_docs = collections.defaultdict(set)
pair_n = collections.Counter()
pair_docs = collections.defaultdict(set)
pair_baskets = collections.defaultdict(list)
for k, items in baskets.items():
    for i in items:
        item_n[i] += 1; item_docs[i].add(basket_doc[k])
    for a, b in itertools.combinations(sorted(items), 2):
        pair_n[(a, b)] += 1
        pair_docs[(a, b)].add(basket_doc[k])
        if len(pair_baskets[(a, b)]) < 30:
            pair_baskets[(a, b)].append(k)

def kind_of(i):
    return "term" if i.startswith("#") else "message" if "-" in i else "program"

def module(i):
    if i.startswith("#"): return "#"
    return i.split("-")[0] if "-" in i else MOD.match(i).group(0)

def evidence(a, b, limit=3):
    """Chunks where both items appear together (best), else chunks of the basket."""
    ev, seen_docs = [], set()
    for k in pair_baskets[(a, b)]:
        if basket_doc[k] in seen_docs: continue       # one example per document
        for cid, items in basket_chunks[k]:
            if a in items and b in items:
                ev.append(cid); seen_docs.add(basket_doc[k]); break
        else:
            ev.append(basket_chunks[k][0][0]); seen_docs.add(basket_doc[k])
        if len(ev) == limit: break
    return ev

rules = []
for (a, b), n in pair_n.items():
    if a.startswith("#") and b.startswith("#") and not args.term_term: continue
    nd = len(pair_docs[(a, b)])
    if n < args.min_together or nd < args.min_docs: continue
    for x, y in ((a, b), (b, a)):
        conf = n / item_n[x]
        py = item_n[y] / N
        lift = conf / py
        if x.startswith("#") and not y.startswith("#"):
            if conf < args.term_confidence or lift < args.term_lift: continue
        elif conf < args.min_confidence or lift < args.min_lift: continue
        conviction = (1 - py) / (1 - conf) if conf < 1 else math.inf
        rules.append({
            "from": x, "to": y, "together": n, "documents": nd,
            "from_baskets": item_n[x], "to_baskets": item_n[y],
            "confidence": round(conf, 3), "lift": round(lift, 2),
            "conviction": round(conviction, 2) if conviction != math.inf else "inf",
            "cross_module": int(module(x) != module(y) and "#" not in (module(x), module(y))),
            "kind": f"{kind_of(x)}->{kind_of(y)}",
            "evidence": " ".join(evidence(a, b)),
        })

# strength used for sorting: confident, above chance, and backed by several documents
# rare items get extreme lifts from a handful of baskets, so the confidence is
# shrunk towards zero when X is rare (adds 3 "virtual" baskets without Y)
for r in rules:
    shrunk = r["together"] / (r["from_baskets"] + 3)
    r["score"] = round(shrunk * math.log2(1 + min(r["lift"], 50)) * math.log2(1 + r["documents"]), 3)
rules.sort(key=lambda r: -r["score"])

with open(os.path.join(args.out, "rules.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rules[0].keys()) if rules else ["from", "to"])
    w.writeheader(); w.writerows(rules)

rules_per_item = collections.Counter(r["from"] for r in rules)
with open(os.path.join(args.out, "items.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["item", "module", "baskets", "documents", "rules_from_item"])
    for i, n in item_n.most_common():
        w.writerow([i, module(i), n, len(item_docs[i]), rules_per_item[i]])

# ------------------------------------------------------------------ report
print(f"unit: {args.unit} | baskets: {N} (from {docs_total} documents; "
      f"{len(too_big)} baskets with > {args.max_items} items ignored; "
      f"{duplicates} copied chunks skipped)")
multi = sum(1 for v in baskets.values() if len(v) >= 2)
print(f"baskets with 2+ items: {multi} ({multi / max(N,1):.0%}) | "
      f"avg items in those: {sum(len(v) for v in baskets.values() if len(v) >= 2) / max(multi,1):.1f}")
print(f"distinct items: {len(item_n)} | items in 5+ baskets: {sum(1 for v in item_n.values() if v >= 5)}")
covered = len({r['from'] for r in rules})
print(f"rules: {len(rules)} | items with at least one rule: {covered} "
      f"({covered / max(len(item_n),1):.0%} of items) | cross-module rules: {sum(r['cross_module'] for r in rules)}")
print("\nstrongest rules:")
print(f"  {'from':10s} {'to':10s} {'tog':>4s} {'docs':>4s} {'conf':>5s} {'lift':>6s}")
for r in rules[:20]:
    print(f"  {r['from']:10s} {r['to']:10s} {r['together']:4d} {r['documents']:4d} "
          f"{r['confidence']:5.2f} {r['lift']:6.1f}")
if find_terms:
    print("\nstrongest rules between terms and programs:")
    for r in [r for r in rules if r["kind"] in ("term->program", "program->term")][:20]:
        print(f"  {r['from']:28s} {r['to']:28s} {r['together']:4d} {r['documents']:4d} "
              f"{r['confidence']:5.2f} {r['lift']:6.1f}")
print("\nstrongest cross-module rules:")
for r in [r for r in rules if r["cross_module"]][:15]:
    print(f"  {r['from']:10s} {r['to']:10s} {r['together']:4d} {r['documents']:4d} "
          f"{r['confidence']:5.2f} {r['lift']:6.1f}")

menu = {}
if args.menu and os.path.exists(args.menu):
    from menu_catalog import load_menu
    menu = load_menu(args.menu)
    print(f"menu catalog: {len(menu)} programs from {args.menu}")

# ------------------------------------------------------------------ html
used = {c for r in rules for c in r["evidence"].split()}
chunks = {r["chunk_id"]: [r.get("title") or r["doc_name"], r["doc_name"],
                          re.sub(r"\[Imagem:[^\]]*\]", "", r["text"])[:700]]
          for r in rows if r["chunk_id"] in used}
names = {}
tn = re.compile(r"^(.*) \(([A-Z]{3,5}\d{1,4})\)$")
for r in rows:
    for part in (r.get("title") or "").split(" — ")[0].split(" + "):
        m = tn.match(part.strip())
        if m: names.setdefault(m.group(2), m.group(1))
names = {c: (menu.get(c, {}).get("name") or n) for c, n in
         {**{c: v for c, v in names.items()}, **{c: m["name"] for c, m in menu.items() if c in item_n}}.items()}
data = {
    "unit": args.unit, "baskets": N,
    "rules": [[r["from"], r["to"], r["together"], r["documents"], r["confidence"], r["lift"],
               r["cross_module"], r["evidence"].split(), r["kind"]] for r in rules],
    "items": {i: [n, len(item_docs[i])] for i, n in item_n.items()},
    "names": names, "chunks": chunks,
}
tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "association_template.html"),
           encoding="utf-8").read()
with open(os.path.join(args.out, "association.html"), "w", encoding="utf-8") as f:
    f.write(tpl.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")))

# ------------------------------------------------------------------ program descriptions
# What each program is about, built from data we already have, so a graph walk
# (and the LLM) can tell that FATU61 is "the NF-e / SEFAZ one".
manual = {}
if args.descriptions and os.path.exists(args.descriptions):
    with open(args.descriptions, encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f, delimiter=";"):
            if len(row) >= 2 and row[0].strip() and row[0].strip().lower() != "program" and row[1].strip():
                manual[row[0].strip().upper()] = row[1].strip()
    print(f"manual descriptions loaded: {len(manual)} from {args.descriptions}")

phrase_count = collections.defaultdict(collections.Counter)
doc_count = collections.defaultdict(collections.Counter)
for r in rows:
    for p in r["main_programs"].split():
        phrase_count[p].update(r.get("key_phrases") or [])
        doc_count[p][re.sub(r"\.(pdf|docx)$", "", r["doc_name"], flags=re.I)] += 1

related_terms = collections.defaultdict(list)
for r in sorted((r for r in rules if r["kind"] == "program->term"),
                key=lambda r: -(r["confidence"] * math.log2(1 + min(r["lift"], 50)))):
    if len(related_terms[r["from"]]) < 6:
        related_terms[r["from"]].append(r["to"][1:])

def describe(code):
    m = menu.get(code, {})
    name = m.get("name") or names.get(code)
    terms = related_terms.get(code, [])
    name_words = set(re.findall(r"\w+", (name or "").lower()))
    phrases = [p for p, n in phrase_count[code].most_common(30)
               if n >= 2 and not set(p.split()) <= name_words and not any(p in t or t in p for t in terms)][:6]
    docs = [d for d, _ in doc_count[code].most_common(3)]
    parts = [f"{code}" + (f" – {name}" if name else "") + "."]
    if m.get("type"): parts.append(f"Tipo: {m['type']}.")
    if code in manual: parts.append(manual[code].rstrip(".") + ".")
    if names.get(code) and name and names[code].lower() not in name.lower():
        parts.append(f"Nos documentos também aparece como: {names[code]}.")
    if m.get("variants"): parts.append("Variações no menu: " + "; ".join(m["variants"][:4]) + ".")
    if terms: parts.append("Assuntos: " + ", ".join(terms) + ".")
    if phrases: parts.append("Termos frequentes: " + ", ".join(phrases) + ".")
    if docs: parts.append("Documentos: " + "; ".join(docs) + ".")
    return {"description": " ".join(parts), "manual_description": manual.get(code),
            "related_terms": ", ".join(terms), "key_phrases": ", ".join(phrases),
            "main_documents": "; ".join(docs), "menu_name": m.get("name"),
            "program_type": m.get("type"), "menu_variants": "; ".join(m.get("variants", [])[:8]),
            "menu_entries": m.get("entries")}

descriptions = {i: describe(i) for i in item_n if kind_of(i) == "program"}
with open(os.path.join(args.out, "program_descriptions_template.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f, delimiter=";")
    w.writerow(["program", "description", "menu name", "type", "auto description (for reference)"])
    for i in sorted(descriptions, key=lambda i: -item_n[i]):
        d = descriptions[i]
        w.writerow([i, manual.get(i, ""), d["menu_name"] or names.get(i, ""), d["program_type"] or "",
                    d["description"]])

# ------------------------------------------------------------------ sqlite
import sqlite3, datetime
db_path = args.db or os.path.join(args.out, "associations.db")
version = args.version or datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
SCHEMA = """
CREATE TABLE IF NOT EXISTS run (
  version      TEXT PRIMARY KEY,         -- label of this run
  created_at   TEXT NOT NULL,
  chunks_file  TEXT NOT NULL,
  unit         TEXT NOT NULL,            -- what a basket is: chunk/topic/record/document
  params_json  TEXT NOT NULL,            -- every threshold used
  baskets      INTEGER, documents INTEGER, items INTEGER, rules INTEGER
);
CREATE TABLE IF NOT EXISTS node (
  version      TEXT NOT NULL REFERENCES run(version) ON DELETE CASCADE,
  node_id      TEXT NOT NULL,            -- 'PEDI4', 'KND-004070', '#nota fiscal'
  node_type    TEXT NOT NULL,            -- program | message | term
  name         TEXT,                     -- learned program name, if any
  module       TEXT,                     -- code prefix ('PEDI'), 'KND'/'ORA', or '#' for terms
  baskets      INTEGER NOT NULL,         -- baskets that contain the item
  documents    INTEGER NOT NULL,         -- distinct documents behind them
  rules_from   INTEGER NOT NULL,         -- rules that start at this item
  description        TEXT,             -- what the program is about (for the LLM)
  manual_description TEXT,             -- from program_descriptions.csv, if given
  related_terms      TEXT,             -- business terms most tied to the program
  key_phrases        TEXT,             -- frequent phrases in its own documents
  main_documents     TEXT,             -- documents mainly about it
  menu_name          TEXT,             -- official name from the ERP menu
  program_type       TEXT,             -- 'Programa' or 'Relatório'
  menu_variants      TEXT,             -- other menu labels (customer versions)
  menu_entries       INTEGER,          -- how many menu entries point to it
  PRIMARY KEY (version, node_id)
);
CREATE TABLE IF NOT EXISTS edge (
  edge_id      INTEGER PRIMARY KEY,
  version      TEXT NOT NULL REFERENCES run(version) ON DELETE CASCADE,
  from_node    TEXT NOT NULL,
  to_node      TEXT NOT NULL,
  relation     TEXT NOT NULL,            -- 'co_mentioned' (future: co_changed, reads_table, caused_failure ...)
  source       TEXT NOT NULL,            -- 'docs_assoc' (future: pedido_changes, oracle_deps, manual ...)
  kind         TEXT NOT NULL,            -- program->program, term->program ...
  together     INTEGER NOT NULL,         -- baskets with both items
  documents    INTEGER NOT NULL,         -- distinct documents with both
  from_baskets INTEGER NOT NULL,
  to_baskets   INTEGER NOT NULL,
  confidence   REAL NOT NULL,            -- together / from_baskets
  lift         REAL NOT NULL,
  conviction   REAL,                     -- NULL = infinite (confidence 1)
  score        REAL NOT NULL,            -- ranking score used in rules.csv
  weight       REAL NOT NULL,            -- 0..1 strength for graph walks: together / (from_baskets + 3)
  cross_module INTEGER NOT NULL,         -- 1 = different program modules
  UNIQUE (version, from_node, to_node, relation, source)
);
CREATE INDEX IF NOT EXISTS edge_from ON edge (version, from_node, weight DESC);
CREATE INDEX IF NOT EXISTS edge_to   ON edge (version, to_node);
CREATE TABLE IF NOT EXISTS edge_evidence (
  edge_id      INTEGER NOT NULL REFERENCES edge(edge_id) ON DELETE CASCADE,
  rank         INTEGER NOT NULL,         -- 1 = first example
  chunk_id     TEXT NOT NULL,
  PRIMARY KEY (edge_id, rank)
);
CREATE TABLE IF NOT EXISTS chunk (         -- only the chunks used as evidence
  version      TEXT NOT NULL REFERENCES run(version) ON DELETE CASCADE,
  chunk_id     TEXT NOT NULL,
  doc_id       INTEGER, doc_name TEXT, title TEXT, record_type TEXT,
  pedido       TEXT, topic_id TEXT, text TEXT NOT NULL,
  PRIMARY KEY (version, chunk_id)
);
CREATE VIEW IF NOT EXISTS edge_full AS
  SELECT e.*, nf.name AS from_name, nt.name AS to_name, nt.module AS to_module,
         (SELECT group_concat(chunk_id, ' ') FROM
            (SELECT chunk_id FROM edge_evidence v WHERE v.edge_id = e.edge_id ORDER BY rank)) AS evidence
  FROM edge e
  LEFT JOIN node nf ON nf.version = e.version AND nf.node_id = e.from_node
  LEFT JOIN node nt ON nt.version = e.version AND nt.node_id = e.to_node;
"""
con = sqlite3.connect(db_path)
con.execute("PRAGMA foreign_keys = ON")
con.executescript(SCHEMA)
# databases created by an older version of this script: add the new columns
have = {r[1] for r in con.execute("PRAGMA table_info(node)")}
for col in ("description", "manual_description", "related_terms", "key_phrases", "main_documents",
            "menu_name", "program_type", "menu_variants", "menu_entries"):
    if col not in have:
        con.execute(f"ALTER TABLE node ADD COLUMN {col} TEXT")
with con:
    con.execute("DELETE FROM run WHERE version = ?", (version,))      # cascades to the other tables
    params = {k: v for k, v in vars(args).items() if k not in ("out", "db", "version")}
    con.execute("INSERT INTO run VALUES (?,?,?,?,?,?,?,?,?)",
                (version, datetime.datetime.now().isoformat(timespec="seconds"),
                 os.path.abspath(args.chunks), args.unit, json.dumps(params, ensure_ascii=False),
                 N, docs_total, len(item_n), len(rules)))
    empty = dict.fromkeys(("description", "manual_description", "related_terms", "key_phrases",
                           "main_documents", "menu_name", "program_type", "menu_variants",
                           "menu_entries"))
    con.executemany(
        "INSERT INTO node (version, node_id, node_type, name, module, baskets, documents, rules_from,"
        " description, manual_description, related_terms, key_phrases, main_documents,"
        " menu_name, program_type, menu_variants, menu_entries)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [(version, i, kind_of(i), menu.get(i, {}).get("name") or names.get(i), module(i), n,
          len(item_docs[i]), rules_per_item[i], *descriptions.get(i, empty).values())
         for i, n in item_n.items()])
    for r in rules:
        cur = con.execute(
            "INSERT INTO edge (version, from_node, to_node, relation, source, kind, together, documents,"
            " from_baskets, to_baskets, confidence, lift, conviction, score, weight, cross_module)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (version, r["from"], r["to"], "co_mentioned", "docs_assoc", r["kind"], r["together"],
             r["documents"], r["from_baskets"], r["to_baskets"], r["confidence"], r["lift"],
             None if r["conviction"] == "inf" else r["conviction"], r["score"],
             round(r["together"] / (r["from_baskets"] + 3), 4), r["cross_module"]))
        con.executemany("INSERT INTO edge_evidence VALUES (?,?,?)",
                        [(cur.lastrowid, k + 1, c) for k, c in enumerate(r["evidence"].split())])
    ev_ids = {c for r in rules for c in r["evidence"].split()}
    con.executemany("INSERT INTO chunk VALUES (?,?,?,?,?,?,?,?,?)",
                    [(version, r["chunk_id"], r["doc_id"], r["doc_name"], r.get("title"),
                      r["record_type"], r["pedido"], r.get("topic_id"), r["text"])
                     for r in rows if r["chunk_id"] in ev_ids])
con.execute("VACUUM")
con.close()
print(f"\nwritten: {args.out}/rules.csv, items.csv, association.html")
print(f"descriptions: {len(descriptions)} programs "
      f"(template to complete by hand: {args.out}/program_descriptions_template.csv)")
print(f"database: {db_path}  (run '{version}': {len(item_n)} nodes, {len(rules)} edges, "
      f"{len(ev_ids)} evidence chunks)")
