"""
Find what else a development could impact, walking the association graph
stored by association.py (associations.db).

    python impact.py PEDI4                         # latest run in assoc_out/associations.db
    python impact.py PEDI4 "#nota fiscal de saída" --hops 2 --limit 15
    python impact.py PEDI4 --json                  # machine-readable, e.g. for the chatbot tool
    python impact.py --list-runs

From Python (e.g. as the chatbot's tool):

    from impact import find_impact_candidates
    result = find_impact_candidates("assoc_out/associations.db", ["PEDI4"], max_hops=2, limit=15)

How the score works
    - each edge has a weight 0..1 (shrunk confidence: together / (from_baskets + 3))
    - a path's strength = product of its edge weights x hop_penalty for every extra hop
    - an item reached by several paths / inputs combines them with noisy-OR:
      score = 1 - (1 - s1)(1 - s2)...
    Only the best path from each input is counted, so one strong chain is not counted twice.
"""
import argparse, json, sqlite3, sys

def latest_version(con):
    row = con.execute("SELECT version FROM run ORDER BY created_at DESC LIMIT 1").fetchone()
    if not row:
        raise ValueError("the database has no runs; run association.py first")
    return row[0]

def normalize(item):
    item = item.strip()
    return "#" + item[1:].strip().lower() if item.startswith("#") else item.upper()

def find_impact_candidates(db_path, items, max_hops=2, limit=15, version=None, min_weight=0.05,
                           hop_penalty=0.5, min_score=0.02, relations=None, include_terms=False,
                           evidence_per_item=2):
    """Return {"version", "inputs", "unknown", "candidates": [...]}.
    Each candidate: item, type, name, module, score, hops, cross_module,
    paths (best path per input, with the relation and weight of each step) and evidence chunks."""
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    version = version or latest_version(con)
    items = [normalize(i) for i in items if i.strip()]
    known = {r["node_id"] for r in con.execute(
        f"SELECT node_id FROM node WHERE version = ? AND node_id IN ({','.join('?' * len(items))})",
        [version, *items])} if items else set()
    rel_sql = f" AND relation IN ({','.join('?' * len(relations))})" if relations else ""

    def edges_from(node):
        return con.execute(
            "SELECT edge_id, to_node, relation, weight, confidence, lift, documents, cross_module "
            "FROM edge WHERE version = ? AND from_node = ? AND weight >= ?" + rel_sql +
            " ORDER BY weight DESC LIMIT 50",
            [version, node, min_weight, *(relations or [])]).fetchall()

    # best path from each input to each reachable node (strongest product of weights)
    best = {}                                        # (input, node) -> (strength, path, steps)
    for src in known:
        frontier = {src: (1.0, [src], [])}
        for hop in range(1, max_hops + 1):
            nxt = {}
            for node, (strength, path, steps) in frontier.items():
                for e in edges_from(node):
                    tgt = e["to_node"]
                    if tgt in path: continue                     # no cycles
                    s = strength * e["weight"] * (hop_penalty if hop > 1 else 1.0)
                    step = {"from": node, "to": tgt, "relation": e["relation"],
                            "weight": e["weight"], "confidence": e["confidence"],
                            "lift": e["lift"], "documents": e["documents"], "edge_id": e["edge_id"],
                            "cross_module": bool(e["cross_module"])}
                    key = (src, tgt)
                    if key not in best or s > best[key][0]:
                        best[key] = (s, path + [tgt], steps + [step])
                    if tgt not in nxt or s > nxt[tgt][0]:
                        nxt[tgt] = (s, path + [tgt], steps + [step])
            # expanding terms spreads everywhere; walk only through programs/messages
            frontier = {n: v for n, v in nxt.items() if not n.startswith("#")}

    # combine the inputs per candidate
    cand = {}
    for (src, tgt), (s, path, steps) in best.items():
        if tgt in known: continue
        if tgt.startswith("#") and not include_terms: continue
        c = cand.setdefault(tgt, {"miss": 1.0, "paths": [], "hops": 99})
        c["miss"] *= (1 - s)
        c["hops"] = min(c["hops"], len(steps))
        c["paths"].append({"input": src, "strength": round(s, 4), "path": path, "steps": steps})

    ranked = sorted(((1 - c["miss"], n, c) for n, c in cand.items()), key=lambda x: -x[0])
    ranked = [x for x in ranked if x[0] >= min_score][:limit]

    out = []
    for score, node, c in ranked:
        info = con.execute("SELECT node_type, name, module FROM node WHERE version = ? AND node_id = ?",
                           (version, node)).fetchone()
        c["paths"].sort(key=lambda p: -p["strength"])
        # evidence: chunks behind the strongest path's last edge
        last_edge = c["paths"][0]["steps"][-1]["edge_id"]
        ev = con.execute(
            "SELECT k.chunk_id, k.doc_name, k.title, k.pedido FROM edge_evidence v "
            "JOIN chunk k ON k.version = ? AND k.chunk_id = v.chunk_id "
            "WHERE v.edge_id = ? ORDER BY v.rank LIMIT ?",
            (version, last_edge, evidence_per_item)).fetchall()
        out.append({
            "item": node, "type": info["node_type"], "name": info["name"], "module": info["module"],
            "score": round(score, 4), "hops": c["hops"],
            "cross_module": any(st["cross_module"] for p in c["paths"] for st in p["steps"]),
            "paths": c["paths"],
            "evidence": [dict(r) for r in ev],
        })
    con.close()
    return {"version": version, "inputs": sorted(known),
            "unknown": [i for i in items if i not in known], "candidates": out}

def chunk_text(db_path, chunk_id, version=None):
    """Full text of an evidence chunk (to send to the LLM)."""
    con = sqlite3.connect(db_path)
    version = version or latest_version(con)
    row = con.execute("SELECT text FROM chunk WHERE version = ? AND chunk_id = ?",
                      (version, chunk_id)).fetchone()
    con.close()
    return row[0] if row else None

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="What else could a change impact?")
    ap.add_argument("items", nargs="*", help="programs, messages or #terms changed by the pedido")
    ap.add_argument("--db", default="assoc_out/associations.db")
    ap.add_argument("--version")
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--min-weight", type=float, default=0.05)
    ap.add_argument("--terms", action="store_true", help="also suggest business terms")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--list-runs", action="store_true")
    a = ap.parse_args()

    if a.list_runs:
        con = sqlite3.connect(a.db)
        for r in con.execute("SELECT version, created_at, unit, baskets, items, rules FROM run ORDER BY created_at"):
            print(f"{r[0]:20s} {r[1]}  unit={r[2]}  baskets={r[3]}  items={r[4]}  rules={r[5]}")
        sys.exit()
    if not a.items:
        ap.error("give at least one item, e.g. PEDI4")
    res = find_impact_candidates(a.db, a.items, a.hops, a.limit, a.version, a.min_weight,
                                 include_terms=a.terms)
    if a.json:
        print(json.dumps(res, ensure_ascii=False, indent=2)); sys.exit()
    print(f"run {res['version']} | inputs: {', '.join(res['inputs']) or '-'}"
          + (f" | not found: {', '.join(res['unknown'])}" if res["unknown"] else ""))
    for c in res["candidates"]:
        p = c["paths"][0]
        route = " -> ".join(p["path"])
        print(f"{c['score']:.2f}  {c['item']:12s} {(c['name'] or '')[:34]:34s} "
              f"{'[other module] ' if c['cross_module'] else ''}via {route}")
        for e in c["evidence"]:
            print(f"        evidence {e['chunk_id']}: {(e['title'] or e['doc_name'])[:80]}")
