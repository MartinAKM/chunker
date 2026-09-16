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
ap.add_argument("--min-together", type=int, default=3)
ap.add_argument("--min-docs", type=int, default=3)
ap.add_argument("--min-confidence", type=float, default=0.2)
ap.add_argument("--min-lift", type=float, default=2.0)
ap.add_argument("--max-items", type=int, default=25,
                help="ignore baskets with more items than this (lists/indexes that mention everything)")
ap.add_argument("--out", default="assoc_out")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

MSG = re.compile(r"\b(?:KND|ORA)-\s?\d+")
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

too_big = [k for k, v in baskets.items() if len(v) > args.max_items]
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

def module(i):
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
    nd = len(pair_docs[(a, b)])
    if n < args.min_together or nd < args.min_docs: continue
    for x, y in ((a, b), (b, a)):
        conf = n / item_n[x]
        py = item_n[y] / N
        lift = conf / py
        if conf < args.min_confidence or lift < args.min_lift: continue
        conviction = (1 - py) / (1 - conf) if conf < 1 else math.inf
        rules.append({
            "from": x, "to": y, "together": n, "documents": nd,
            "from_baskets": item_n[x], "to_baskets": item_n[y],
            "confidence": round(conf, 3), "lift": round(lift, 2),
            "conviction": round(conviction, 2) if conviction != math.inf else "inf",
            "cross_module": int(module(x) != module(y)),
            "kind": ("program->program" if "-" not in x + y else
                     "message->program" if "-" in x and "-" not in y else
                     "program->message" if "-" in y and "-" not in x else "message->message"),
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
print("\nstrongest cross-module rules:")
for r in [r for r in rules if r["cross_module"]][:15]:
    print(f"  {r['from']:10s} {r['to']:10s} {r['together']:4d} {r['documents']:4d} "
          f"{r['confidence']:5.2f} {r['lift']:6.1f}")

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
data = {
    "unit": args.unit, "baskets": N,
    "rules": [[r["from"], r["to"], r["together"], r["documents"], r["confidence"], r["lift"],
               r["cross_module"], r["evidence"].split()] for r in rules],
    "items": {i: [n, len(item_docs[i])] for i, n in item_n.items()},
    "names": names, "chunks": chunks,
}
tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "association_template.html"),
           encoding="utf-8").read()
with open(os.path.join(args.out, "association.html"), "w", encoding="utf-8") as f:
    f.write(tpl.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")))
print(f"\nwritten: {args.out}/rules.csv, items.csv, association.html")
