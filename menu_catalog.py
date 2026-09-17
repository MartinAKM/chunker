"""
Read the ERP menu export (DADOS_MENU.csv: PROGRAMA;DESCRICAO;TIPO) and turn it
into one entry per program.

    from menu_catalog import load_menu
    menu = load_menu("DADOS_MENU.csv")
    menu["FATU61"]["name"]      -> 'Monitor Notas Fiscais Eletrônicas'
    menu["FATU61"]["type"]      -> 'Programa' or 'Relatório'
    menu["FATU61"]["variants"]  -> other menu labels (customer versions, banks, ...)
    menu["FATU61"]["entries"]   -> how many menu entries point to the program

The same program appears many times in the menu, once per customer or variation
("06 - Fab - Monitor Notas Fiscais Eletrônicas Romaneio", "Monitor Notas Fiscais
Eletrônicas - Equador"). The common name is the longest beginning shared by a
good share of those labels, so the customisations drop off.

Codes with a suffix (FATU15_BRISA, PEDI6_MAX) count for the base program
(FATU15, PEDI6) and are also listed in "customised".
The file is usually saved as latin-1 by Oracle tools; utf-8 is tried first.
"""
import collections, csv, io, re

NUM_PREFIX = re.compile(r"^\s*(?:\d{1,2}\s*[-.)]\s*|\*{2,}[^*]*\*{2,}\s*|[-–]\s*)+")
SPACES = re.compile(r"\s+")

def clean_label(s):
    s = SPACES.sub(" ", (s or "").replace("\xa0", " ")).strip(" -–\t")
    prev = None
    while prev != s:                       # "06 - Fab - Monitor ..." -> "Fab - Monitor ..."
        prev = s
        s = NUM_PREFIX.sub("", s).strip(" -–")
    return s

def common_name(labels):
    """Longest beginning (in words) shared by at least 40% of the labels."""
    lists = [l.split() for l in labels if l]
    if not lists:
        return ""
    need = max(2, int(len(lists) * 0.4))
    best, i = "", 0
    while True:
        words = collections.Counter(w[i].lower() for w in lists if len(w) > i)
        if not words:
            break
        word, n = words.most_common(1)[0]
        if n < need:
            break
        lists = [w for w in lists if len(w) > i and w[i].lower() == word]
        i += 1
        best = " ".join(lists[0][:i])
    best = best.strip(" -–(,")
    if len(best) < 4:                      # nothing in common: use the most frequent label
        best = collections.Counter(labels).most_common(1)[0][0]
    return best

def load_menu(path):
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "latin-1"):
        try:
            text = raw.decode(enc); break
        except UnicodeDecodeError:
            continue
    sample = text[:2000]
    delim = ";" if sample.count(";") > sample.count(",") else ","
    rows = list(csv.DictReader(io.StringIO(text), delimiter=delim))
    cols = {c.lower().strip(): c for c in (rows[0].keys() if rows else {})}
    c_prog = cols.get("programa") or list(cols.values())[0]
    c_desc = cols.get("descricao") or cols.get("descrição") or list(cols.values())[1]
    c_type = cols.get("tipo")

    labels, types, custom, count = (collections.defaultdict(list), collections.defaultdict(collections.Counter),
                                    collections.defaultdict(set), collections.Counter())
    for r in rows:
        code = (r.get(c_prog) or "").strip().upper()
        if not code: continue
        base = code.split("_")[0]
        if not re.fullmatch(r"[A-Z]{2,6}\d{1,4}", base): continue
        label = clean_label(r.get(c_desc))
        if label: labels[base].append(label)
        if c_type and r.get(c_type): types[base][r[c_type].strip()] += 1
        if code != base: custom[base].add(code)
        count[base] += 1

    menu = {}
    for code, labs in labels.items():
        name = common_name(labs)
        seen, variants = {name.lower()}, []
        for l, _ in collections.Counter(labs).most_common():
            if l.lower() in seen or l.lower() == name.lower(): continue
            seen.add(l.lower()); variants.append(l)
        menu[code] = {
            "name": name,
            "type": types[code].most_common(1)[0][0] if types.get(code) else None,
            "variants": variants,
            "customised": sorted(custom[code]),
            "entries": count[code],
        }
    return menu

if __name__ == "__main__":
    import sys
    m = load_menu(sys.argv[1] if len(sys.argv) > 1 else "DADOS_MENU.csv")
    print(f"{len(m)} programs")
    for code in (sys.argv[2:] or ["FATU2", "FATU61", "EPRO15", "PEDI4", "CREC22"]):
        e = m.get(code.upper())
        print(f"\n{code}: {e['name']} [{e['type']}] · {e['entries']} menu entries")
        for v in e["variants"][:5]: print("   variant:", v)
