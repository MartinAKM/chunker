"""
Give every chunk a readable title WITHOUT an LLM.

    python titler.py chunks.jsonl            -> writes chunks_titled.jsonl
    python titler.py chunks.jsonl out.jsonl --menu=DADOS_MENU.csv

Program names come from the ERP menu export when it is available
(DADOS_MENU.csv), and are otherwise learned from the documents.

Title = "<program name (CODE)> — <subject>"

Program name : learned from the corpus itself. Your texts write programs as
               "Estruturação de Produto-EPRO15" or "CFAB17 - Emissão Cartão
               Componente"; the most frequent name per code wins.
Subject      : first source that works, in this order
               1. the request of a pedido ("Solicitação: ...")
               2. a short heading-like first line
               3. key phrases: word sequences that are frequent in this chunk
                  but rare in the rest of the corpus (TF-IDF), ignoring
                  stop words and the ERP screen boilerplate
                  ("tecle F9 ou clique no botão Lista ...").
Only the Python standard library is used.
"""
import json, os, re, sys, math, collections

argv = [a for a in sys.argv[1:] if not a.startswith("--")]
SRC = argv[0] if argv else "chunks.jsonl"
DST = argv[1] if len(argv) > 1 else "chunks_titled.jsonl"
MENU = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--menu=")), "DADOS_MENU.csv")
MAX_SUBJECT = 90

MODULES = "FATU|ESTO|PEDI|ADMI|EPRO|CFAB|MPSP|COMP|VEND|PREC|CEXC|RECE|MIND|CEXT|CPRO|CPAG|FVEN|TERC|CREC|CUST|FPAG|CONT|RHUM|CLAB|LFIS"
UP = "A-ZÁÉÍÓÚÂÊÔÃÕÇ"
NAME_BEFORE = re.compile(rf"([{UP}][\wÀ-ú/ ]{{3,50}}?)\s?[-–]\s?((?:{MODULES})\d{{1,4}})\b")
NAME_AFTER  = re.compile(rf"\b((?:{MODULES})\d{{1,4}})\s?[-–]\s?([{UP}][\wÀ-ú/ ]{{3,50}})")

STOP = set("""
a à ao aos as às até após com como da das de do dos e é em entre era essa esse esta este estes
estas eu foi foram há isso isto já la lhe mais mas me mesmo na nas nem no nos não o os ou para
pela pelas pelo pelos por qual quando que quem se sem ser seu seus sua suas são só também te tem
ter um uma umas uns você vai será sendo estar está estão pode podem deve devem deverá foi sobre
cada todo toda todos todas outro outra outros outras onde então ainda assim caso bem nesta neste
nesse nessa desta deste dessa desse aqui ali lá seja sejam fica ficar feito feita fazer faz dia
el la los las del por con para una uno que en y o se al lo su sus es
""".split())
# words that appear in almost every procedure and say nothing about the subject
BOILER = set("""
campo campos tecle clique clicar botão botões lista listas valores valor tela telas programa
programas posicionar posicionar-se informar informe localizar desejado desejada desejados
efetuar clics clic sobre selecionar selecionada selecionado abrir aberta aberto fechar voltará
inicial aba abas opção opções registro registros salvar sugerido sugerida sugestão pedido
serviço solicitação detalhamento observação imagem conforme abaixo acima segue seguinte teste
testes testado ok sim cliente favor obrigado obrigada att atenciosamente kunden marcio
tendo informação código códigos diretamente modo entrar navegue navegar barra ferramentas
superior inferior esquerda direita param confirmar consulta consultar executar retornar
automaticamente caso contrário processo após antes agora abaixo necessário conseguir
""".split())
FIELD = re.compile(rf"\b(?:campo|aba|botão|check ?box|opção)\s+[\"“]?([{UP}][\wÀ-ú/()\-]*(?:\s+(?:de|da|do|dos|das|e)?\s*[{UP}(][\wÀ-ú/()\-]*){{0,4}})", re.U)

WORD = re.compile(r"[A-Za-zÀ-ú][A-Za-zÀ-ú]*(?:-[A-Za-zÀ-ú]+)*")

def fold(w): return w.lower()

def usable(w):
    lw = fold(w)
    return len(lw) > 2 and lw not in STOP and lw not in BOILER and not re.search(r"\d", w)

def candidates(text):
    """1-3 word phrases that neither start nor end with a stop/boilerplate word."""
    text = re.sub(r"\[(?:Imagem|Código):[^\]]*\]", " ", text)
    text = re.sub(rf"\b(?:{MODULES})\d{{1,4}}\b", " ", text)
    out = []
    for sent in re.split(r"[.;:!?\n()\"“”*•]", text):
        words = WORD.findall(sent)
        for n in (1, 2, 3):
            for i in range(len(words) - n + 1):
                g = words[i:i + n]
                if not usable(g[0]) or not usable(g[-1]): continue
                if n == 3 and fold(g[1]) not in STOP and not usable(g[1]): continue
                out.append(" ".join(fold(x) for x in g))
    return out

# ------------------------------------------------------------------ load
rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]

# program names learned from the corpus
names = collections.defaultdict(collections.Counter)
for r in rows:
    for m in NAME_BEFORE.finditer(r["text"]):
        names[m.group(2)][m.group(1)] += 1
    for m in NAME_AFTER.finditer(r["text"]):
        names[m.group(1)][m.group(2)] += 1

def clean_name(n):
    n = re.sub(r"^(?:Abrir o |Abra o |No |O )?(?:programa|Programa|PROGRAMA)\s+", "", n.strip())
    return n.strip(" -–")

program_name = {}
menu_names = {}
if MENU and os.path.exists(MENU):
    from menu_catalog import load_menu
    menu_names = {c: m["name"] for c, m in load_menu(MENU).items() if m["name"]}
    print(f"menu catalog: {len(menu_names)} official program names from {MENU}")
for code, cnt in names.items():
    agg = collections.Counter()
    for n, k in cnt.items():
        n = clean_name(n)
        if 4 <= len(n) <= 45 and not n.isupper():   # all-caps are usually complaint titles
            agg[n] += k
    if agg:
        best, k = agg.most_common(1)[0]
        if k >= 2 or len(agg) == 1:
            program_name[code] = best

# document frequency for TF-IDF
df = collections.Counter()
cands = []
for r in rows:
    c = candidates(r["text"])
    cands.append(c)
    df.update(set(c))
N = len(rows)

def key_phrases(i, k=3):
    tf = collections.Counter(cands[i])
    scored = []
    for p, f in tf.items():
        if df[p] < 2 or df[p] > N * 0.15: continue          # typos / generic words
        n = p.count(" ") + 1
        scored.append(((1 + math.log(f)) * math.log(N / df[p]) * (1 + 0.5 * (n - 1)), p))
    picked = []
    for _, p in sorted(scored, reverse=True):
        if any(set(p.split()) & set(q.split()) for q in picked): continue  # no shared words
        picked.append(p)
        if len(picked) == k: break
    return picked

def shorten(s, n=MAX_SUBJECT):
    s = re.sub(r"\s+", " ", s).strip(" .:-–")
    if len(s) <= n: return s
    cut = s[:n].rsplit(" ", 1)[0]
    return cut + "…"

def fields(text):
    """Screen fields a procedure works with: 'campo Classificação Fiscal', 'aba Fiscal'."""
    seen = []
    for m in FIELD.finditer(text):
        f = m.group(1).strip(" -\"”")
        if f.lower() not in {x.lower() for x in seen} and f.lower() not in BOILER:
            seen.append(f)
    return seen

def email_subject(text):
    m = re.search(r"^(?:E-mail — )?Assunto:\s*(.+)$", text, re.M)
    if not m: return ""
    return re.sub(r"^(?:(?:RE|RES|ENC|FW|FWD|TR)\s*:\s*)+", "", m.group(1).strip(), flags=re.I)

def heading(text):
    text = re.sub(r"\[(?:Imagem|Código):[^\]]*\]", "", text)
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    lines = [l for l in lines if not re.match(r"^(\(\.\.\.\)|Pedido\s+(de\s+)?Servi|De:|E-mail —|Enviad|Para:|Cc:)", l, re.I)]
    if not lines: return ""
    first = lines[0]
    first = re.sub(r"^\(\.\.\.\)\s*", "", first)
    if 8 <= len(first) <= 80 and not first.endswith((".", ",", ";")) and "[Imagem" not in first \
            and len(WORD.findall(first)) >= 2:
        return first
    return ""

def subject(i, r):
    t = r["text"]
    if r["record_type"] == "pedido":
        m = re.search(r"Solicita[çc][ãa]o\s*:\s*(.+)", t)
        if m and len(m.group(1).strip()) > 10:
            return shorten(m.group(1)), "solicitacao"
    if r["record_type"] == "email":
        e = email_subject(t)
        if e: return shorten(e), "assunto do email"
    fl = fields(t)
    if len(fl) >= 3 and len(re.findall(r"\bcampo\b", t)) >= 3:
        return shorten("campos: " + ", ".join(fl[:5])), "campos da tela"
    h = heading(t)
    if h and not r.get("topic_number"):
        return shorten(h), "cabecalho"
    kp = key_phrases(i)
    if kp:
        return ", ".join(kp).strip(" -"), "palavras-chave"
    rest = re.sub(r"\[Imagem:[^\]]*\]", " ", t).strip()
    return shorten(h or rest[:80] or "(somente imagens)"), "inicio do texto"

def name_of(c):
    return menu_names.get(c) or program_name.get(c)

def program_part(r):
    codes = r["main_programs"].split() or r["mentioned_programs"].split()[:1]
    parts = [f"{name_of(c)} ({c})" if name_of(c) else c for c in codes[:2]]
    return " + ".join(parts)

with open(DST, "w", encoding="utf-8") as f:
    how = collections.Counter()
    for i, r in enumerate(rows):
        subj, src = subject(i, r)
        prog = program_part(r)
        r["title"] = f"{prog} — {subj}" if prog else subj
        r["title_source"] = src
        r["key_phrases"] = key_phrases(i, 5)
        how[src] += 1
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"{len(rows)} chunks titled -> {DST}")
print("program names learned:", len(program_name))
print("title source:", dict(how))
