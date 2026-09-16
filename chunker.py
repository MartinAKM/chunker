"""
Structure-aware chunker for the ERP documentation export.

0. Clean      : drop useless image captions and remove the text that the PDF
                extraction repeats at every page break.
1. Records    : a document is a log of independent entries. Split on XXXX
                separators, "Pedido Serviço" headers and e-mail headers ("De:").
2. Topics     : inside a record, find the numbered outline (1. / 1.1. / 1.1.1.),
                even when the numbers are in the middle of a line. Each top-level
                item ("2. Abrir o programa ...-ADMI34") is a topic, and a chunk
                NEVER crosses a topic boundary. Records without an outline are a
                single topic.
3. Packing    : inside a topic, split on sub-items / labels / paragraphs, glue
                small pieces up to MAX_CHARS, split oversized ones by sentence.
                Every chunk gets a context header (document, topic, programs).
"""
import json, re, sys

MAX_CHARS = 2400      # ~600 tokens for Portuguese text
TOPIC_MAX = 6000      # a numbered topic up to this size stays in ONE chunk (~1500 tokens)
MIN_CHARS = 400       # smaller pieces are merged with neighbours (same topic only)
OVERLAP   = 200       # chars repeated from the previous chunk (same topic only)

MODULES = {"FATU","ESTO","PEDI","ADMI","EPRO","CFAB","MPSP","COMP","VEND","PREC","CEXC","RECE",
           "MIND","CEXT","CPRO","CPAG","FVEN","TERC","CREC","CUST","FPAG","CONT","RHUM","CLAB","LFIS"}
CODE = re.compile(r"\b([A-Z]{3,5})(\d{1,4})\b")

SEP        = re.compile(r"X{15,}x?")          # separator, often glued to the next line
# a real pedido header is "Pedido Serviço: 123", "Pedido Serviço 123:" or
# "Pedido Serviço 123 Solicitação/Cliente" -- a sentence that merely starts with
# "pedido serviço 69438 poderá ..." is NOT a new record
REC_START  = re.compile(r"^(?=(?:Desenvolvido\s+(?:pelo\s+)?)?Pedido\s+(?:de\s+)?Servi[çc]o"
                        r"(?:\s*:\s*\d+|\s+\d+\s*(?::|Solicita|Cliente))"
                        r"|De:\s.+|E-mail —|Testes feitos)", re.M | re.I)
LABEL      = re.compile(r"^(?=(?:Solicita[çc][ãa]o|Solicitado|Detalhamento|Observa[çc][ãa]o|"
                        r"Situa[çc][õo]es|Rotina|Documenta[çc][ãa]o)\s*:)", re.M | re.I)
# outline number: at start of text or after whitespace, followed by a capital letter / quote
# "1. Texto", "1.2) Texto" and the step style "1= texto" / "8 = texto"
OUTLINE    = re.compile(r"(?:(?<=\s)|^)(\d{1,2}(?:\.\d{1,2}){0,4})(?:[.)]\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ\"“])|\s?=\s*(?=[^\s=\d]))")
# error/info message: code, plus its text only when introduced by ":" or " - "
MSG        = re.compile(r"\b((?:KND|ORA)-\s?\d+)(?:\s*(?::|\s-)\s*([^\n\]]{1,200}?)(?=\s*(?:\n|\]|\d\.\s|$)))?")
PEDIDO     = re.compile(r"Pedido\s+(?:de\s+)?Servi[çc]o:?\s*(\d+)", re.I)
SENT       = re.compile(r"(?<=[.!?:;])\s+")

def codes(t):
    return sorted({f"{p}{n}" for p, n in CODE.findall(t) if p in MODULES})

# ---------------------------------------------------------------- 0. cleaning
SEP_ONLY = re.compile(r"^\s*X{15,}x?\s*$")

def find_repeat(content, nxt):
    """If `nxt` starts by repeating the end of `content` (possibly starting in
    the middle of a word), return the index in `nxt` where new text begins."""
    tail = content[-700:]
    for s in (0, nxt.find(" ") + 1 if " " in nxt[:30] else -1):
        if s < 0: continue
        anchor = nxt[s:s + 30]
        if len(anchor) < 25: continue
        a = tail.rfind(anchor)
        if a < 0: continue
        rep = tail[a:]
        if nxt[s:s + len(rep)] == rep:
            return s + len(rep)
    return None

def drop_page_repeats(t):
    """At page breaks the PDF extraction repeats the last ~50-300 chars of the
    previous page, starting mid-word. The repeat itself may contain blank lines,
    and sometimes a separator line (XXXX) or a cut-off header fragment
    ("De: Volnei Antonio Machado (Kund", "Pedid") sits before it.
    Remove the repeat and such fragments; keep separators."""
    parts = t.split("\n\n")
    out = content = parts[0]
    after_sep = False
    i = 1
    while i < len(parts):
        nxt = parts[i]
        if SEP_ONLY.match(nxt):
            out += "\n\n" + nxt; after_sep = True; i += 1
            continue
        rest = "\n\n".join(parts[i:i + 12])       # the repeat may span blank lines
        cut = find_repeat(content, rest)
        skip = 0
        if cut is None and i + 1 < len(parts) and len(nxt) < 60 \
                and not nxt.rstrip().endswith((".", "!", "?")):
            # short cut-off fragment followed by the repeat -> drop the fragment
            rest2 = "\n\n".join(parts[i + 1:i + 13])
            cut2 = find_repeat(content, rest2)
            if cut2 is not None:
                skip, cut, rest = 1, cut2, rest2
        if cut is None:
            out += "\n\n" + nxt; content += "\n\n" + nxt; after_sep = False
            i += 1
            continue
        # consume the parts covered by the repeat; keep what follows it
        consumed = rest[:cut]
        n_parts = consumed.count("\n\n")
        tail = rest[cut:].split("\n\n")[0]          # remainder of the last covered part
        if not after_sep:
            out += tail
        elif tail.strip():
            out += "\n\n" + tail.lstrip()
        content += tail
        if tail.strip(): after_sep = False
        i += skip + n_parts + 1
    return out, 0

CAP_MAX = 8000   # an unclosed caption never swallows more than this

def find_captions(t):
    """Yield (start, end) of every '[Imagem: ...]' block. Captions can contain
    blank lines and nested brackets ("Gerenciador Serviços [193.2]"), so the
    end is the matching ']' by bracket depth. An unclosed caption ends at the
    next '[Imagem:' (or after CAP_MAX chars / end of text)."""
    pos = 0
    while True:
        a = t.find("[Imagem:", pos)
        if a < 0: return
        nxt = t.find("[Imagem:", a + 8)
        limit = min(x for x in (nxt if nxt >= 0 else len(t), a + CAP_MAX, len(t)))
        depth, end = 0, None
        for i in range(a, limit):
            ch = t[i]
            if ch == "[": depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0: end = i + 1; break
        if end is None:
            end = limit
        yield a, end
        pos = end

def strip_captions(t):
    out, last = [], 0
    for a, b in find_captions(t):
        out.append(t[last:a]); out.append(caption(t[a:b])); last = b
    out.append(t[last:])
    return "".join(out)

def caption(cap):
    """Keep only what matters from an image caption: program codes and
    KND-/ORA- messages. Everything else (screen descriptions) is dropped."""
    progs = codes(cap)
    msgs = []
    for code, txt in MSG.findall(cap):
        item = re.sub(r"\s", "", code) + (f": {txt.strip()}" if txt else "")
        if item not in msgs: msgs.append(item)
    if not progs and not msgs:
        return ""
    return "[Imagem: " + " | ".join(progs + msgs) + "]"

def drop_orphan_caption_tails(t):
    """Page breaks cut some captions in two: the tail ("...texto cinza.]") ends
    with ']' but has no '['. Remove such tails and loose description lines."""
    out = []
    for para in t.split("\n\n"):
        close = para.find("]")
        opn = para.find("[")
        if close >= 0 and (opn < 0 or close < opn) and close < 1500:
            para = para[close + 1:]
        para = re.sub(r"(?m)^[.\s]*(?:\d\.\s*)?(?:A imagem|Nenhum código)\b.*$", "", para)
        if para.strip():
            out.append(para.strip())
    return "\n\n".join(out)


# ---------------------------------------------------------------- e-mail noise
EMAIL_FIELDS = r"(?:De|From|Enviada em|Enviado em|Enviado el|Enviado|Sent|Date|Data|Para|To|Cc|CC|Cco|Bcc|Importância|Importance|Assunto|Subject|Asunto)"
EMAIL_HDR = re.compile(
    r"^[ \t]*(?:De|From)[ \t]*:[^\n]*?(?:\n[^\n]*?){0,6}?"          # the From line (+ up to 6 more lines)
    r"(?:Assunto|Subject|Asunto)[ \t]*:[ \t]*([^\n]*)",              # ... until the Subject; keep it
    re.M)
EMAIL_LEFTOVER = re.compile(rf"^[ \t]*{EMAIL_FIELDS}[ \t]*:.*$", re.M)
SUBJ_PREFIX = re.compile(r"^(?:(?:RE|RES|ENC|FW|FWD|TR|RV)\s*:\s*)+", re.I)

def clean_email(t):
    """Replace every e-mail header block with one line 'E-mail — Assunto: ...'
    (the subject says what the e-mail is about; names, addresses and dates don't),
    and remove forward markers, quoted-reply lines, greetings and signatures."""
    def hdr(m):
        subj = SUBJ_PREFIX.sub("", m.group(1).strip())
        block = m.group(0)
        # the header must really look like one: From plus Sent/To/Cc
        if not re.search(r"(?:Enviad[ao]|Sent|Para|To|Cc|Date|Data)\s*:", block):
            return block
        return f"E-mail — Assunto: {subj}" if subj else "E-mail —"
    t = EMAIL_HDR.sub(hdr, t)
    # From lines without a subject, directly followed by header fields
    t = re.sub(rf"^[ \t]*(?:De|From)[ \t]*:[^\n]*\n(?:[ \t]*{EMAIL_FIELDS}[ \t]*:[^\n]*\n?)+",
               "E-mail —\n", t, flags=re.M)
    t = re.sub(r"\[mailto:[^\]]*\]", "", t)
    t = re.sub(r"^[ \t]*-{3,}\s*(?:Mensagem encaminhada|Forwarded message|Mensaje reenviado|Original Message|Mensagem original)\s*-{3,}[ \t]*", "", t, flags=re.M | re.I)
    t = re.sub(r"^[ \t]*(?:Em|On|El)\s.{5,80}?(?:escreveu|wrote|escribió)\s*:[ \t]*$", "", t, flags=re.M | re.I)
    # greetings alone on a line
    t = re.sub(r"^[ \t]*(?:Bom dia|Boa tarde|Boa noite|Olá|Ola|Oi|Prezad[oa]s?|Buen(?:os)? d[ií]as?|Buenas tardes|Hola|Hi|Hello)\b[^\n.]{0,40}[,!]?[ \t]*$",
               "", t, flags=re.M | re.I)
    # sign-off + up to 5 short signature lines (name, role, company, phone, site)
    t = re.sub(r"^[ \t]*(?:Att\.?|Atte\.?|Atenciosamente|Abra[çc]os?|Abs\.?|Saludos(?: cordiales)?|Cordialmente|Obrigad[oa]|Grat[oa]|Regards|Best regards)[,.!]?[ \t]*\n"
               r"(?:[ \t]*[^\n.:;]{1,60}[ \t]*(?:\n|$)){0,5}",
               "\n", t, flags=re.M | re.I)
    t = re.sub(r"^[ \t]*(?:Tel|Fone|Telefone|Cel|Celular|Phone|Whats(?:app)?)\.?[ \t]*:?[ \t]*[+()\d][\d ()+.-]{6,}.*$", "", t, flags=re.M | re.I)
    t = re.sub(r"^[ \t]*(?:www\.\S+|\S+@\S+\.\S+)[ \t]*$", "", t, flags=re.M)
    return t

# ---------------------------------------------------------------- code removal
CODE_START = re.compile(
    r"\b(?:SELECT\b[^\n]*?\bFROM\b|SELECT\s*$|INSERT\s+INTO\b|UPDATE\s+[\w.]+\s*(?:\w+\s*)?(?:SET\b|$)|DELETE\s+FROM\b|"
    r"DECLARE\b|CREATE\s+OR\s+REPLACE\b|CREATE\s+(?:TABLE|INDEX|VIEW)\b|ALTER\s+TABLE\b|CURSOR\s+\w+\s+IS\b|"
    r"BEGIN\s*$|PROCEDURE\s+\w+\s*[(;]|FUNCTION\s+\w+\s*\()",
    re.I)
CODE_CONT = re.compile(
    r"^\s*(?:WHERE|AND|OR|FROM|JOIN|INNER|LEFT|RIGHT|OUTER|FULL|ON|GROUP\s+BY|ORDER\s+BY|HAVING|CONNECT\s+BY|"
    r"START\s+WITH|SET|VALUES|INTO|OPEN|FETCH|CLOSE|LOOP|EXIT|END|IF|ELSE|ELSIF|THEN|COMMIT|ROLLBACK|EXCEPTION|"
    r"BEGIN|RETURN|UNION|CASE|WHEN|SELECT|NVL|DECODE|TO_CHAR|TO_DATE|SUM|COUNT|MAX|MIN|DISTINCT|AS|IS|NOT|"
    r"EXISTS|IN|NULL|FOR|WHILE|RAISE\w*|DBMS_\w+|--|/\*|\*/|\)|\(|,)(?=\W|$)", re.I)
TABLE_REF = re.compile(r"\b(?:FROM|JOIN|INTO|UPDATE|TABLE)\s+([A-Za-z_][\w$#]*(?:\.[A-Za-z_][\w$#]*)?)", re.I)
SQL_WORDS = {"select", "set", "where", "dual", "values", "the", "exists", "table", "c_dados"}

def looks_like_code(line):
    l = line.strip()
    if not l: return False
    if CODE_CONT.match(l) or CODE_START.search(l): return True
    if l.endswith(";") and re.search(r"[=():]", l): return True
    if ":=" in l or re.search(r"%(?:ROWTYPE|TYPE|NOTFOUND|FOUND)\b", l, re.I): return True
    letters = sum(c.isalpha() for c in l)
    return len(l) > 8 and letters / len(l) < 0.55 and not re.search(r"[a-zà-ú]{4,} [a-zà-ú]{4,} [a-zà-ú]{4,}", l)

def looks_like_prose(line):
    words = re.findall(r"[a-zà-ú]{3,}", line)
    return len(words) >= 5 and not line.rstrip().endswith(";") and not CODE_CONT.match(line)

def strip_code(t):
    """Remove SQL / PL-SQL blocks. The code can start in the middle of a line
    ("--exemplo SELECT DISTINCT ... FROM PRODUTO_ATM P"). Each removed block
    leaves a marker with the tables it used: [Código: PRODUTO_ATM]."""
    out, block, in_code = [], [], False
    imgs = []                      # image references found inside code lines are kept
    def flush():
        nonlocal block, imgs
        if imgs:
            out.append(" ".join(imgs)); imgs = []
        if block:
            code = "\n".join(block)
            tabs = []
            for m in TABLE_REF.finditer(code):
                name = m.group(1).upper()
                if name.lower() not in SQL_WORDS and not name.startswith(("V_", "C_")) and name not in tabs:
                    tabs.append(name)
            if tabs:
                out.append("[Código: " + " ".join(tabs[:8]) + "]")
        block = []
    raw = t.split("\n")
    found = [re.findall(r"\[Imagem:[^\]]*\]", l) for l in raw]
    lines = [re.sub(r"\s*\[Imagem:[^\]]*\]", "", l) for l in raw]
    for i, line in enumerate(lines):
        if not in_code:
            nxt0 = lines[i + 1] if i + 1 < len(lines) else ""
            assign = (":=" in line or line.rstrip().endswith(";")) and looks_like_code(line) \
                and not looks_like_prose(line) and looks_like_code(nxt0) and not looks_like_prose(nxt0)
            if assign:                    # PL/SQL statements without a SELECT/DECLARE start
                in_code = True; block = [line]; imgs += found[i]
                continue
            if not CODE_START.search(line) and not re.match(r"\s*SELECT\b", line, re.I):
                out.append(raw[i]); continue
            m = CODE_START.search(line)
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            multi = False
            if re.match(r"\s*SELECT\b(?!\s+(?:de|do|da|deste|desta|para|que|com)\b)", line, re.I) and \
                    any(re.search(r"\bFROM\b", x, re.I) for x in lines[i:i + 10]):
                m = re.match(r"\s*(SELECT\b)", line, re.I)   # SELECT list spread over lines
                m = type("M", (), {"start": lambda self, _p=m.start(1): _p})()
                multi = True
            if m and (multi or looks_like_code(nxt) or line.rstrip().endswith(";") or
                      re.search(r"\bFROM\b", line[m.start():], re.I)):
                before = line[:m.start()].rstrip()
                before = re.sub(r"(?:--|/\*)[^\n]*$", "", before).rstrip()   # a comment right before the code
                if before: out.append(before)
                in_code = True
                block = [line[m.start():]]
                imgs += found[i]
                continue
            out.append(raw[i])
        else:
            if not line.strip():
                # blank line: stay in code only if the next non-blank line is code
                j = i + 1
                while j < len(lines) and not lines[j].strip(): j += 1
                if j < len(lines) and not looks_like_prose(lines[j]) and \
                        (looks_like_code(lines[j]) or len(block) == 1):
                    continue
                flush(); in_code = False; out.append(raw[i])
            elif not looks_like_prose(line) and (looks_like_code(line) or
                    (len(line) < 120 and not line.rstrip().endswith((".", ":", "!", "?")))):
                block.append(line); imgs += found[i]
            else:
                flush(); in_code = False; out.append(raw[i])
    flush()
    return "\n".join(out)

def clean(t):
    t = t.replace("\r", "")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n[ \t]*\n\s*", "\n\n", t)
    t, _ = drop_page_repeats(t)          # first, so captions are whole again
    t = strip_captions(t)
    t = drop_orphan_caption_tails(t)
    t = clean_email(t)
    t = strip_code(t)
    t = re.sub(r"(?im)^[ \t]*z{8,}[^\n]*?(?=\S|$)", "", t)     # "Zzzzzzzz" continuation markers
    t = re.sub(r"\s*\n\s*(\[Código: [^\]]*\])", r" \1", t)      # code markers travel with the text
    t = re.sub(r"(\[Imagem: [^\]]*\])(\s*\1)+", r"\1", t)   # repeated identical captions
    # glue every image reference to the text before it, so a screenshot never
    # forms a paragraph/section of its own and always travels with real text
    t = re.sub(r"\s*\n\s*(\[Imagem: [^\]]*\])", r" \1", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

# ---------------------------------------------------------------- helpers
def split_at(text, positions):
    idx = sorted({0, *positions})
    return [text[a:b].strip() for a, b in zip(idx, idx[1:] + [len(text)]) if text[a:b].strip()]

def split_on(pattern, text):
    return split_at(text, [m.start() for m in pattern.finditer(text)])

def outline(text):
    """Return (top_level_starts, sub_level_starts). Top-level numbers must run
    1, 2, 3... in order; sub-items must belong to the current top-level item.
    Numbers inside image captions are ignored."""
    masked = text
    for a, b in list(find_captions(text)):
        masked = masked[:a] + " " * (b - a) + masked[b:]
    # table-of-contents lines ("DANFE ........ 95") are not topics
    masked = re.sub(r"^.*\.{5,}.*$", lambda m: " " * len(m.group(0)), masked, flags=re.M)
    tops, subs, expect, cur = [], [], 1, 0
    for m in OUTLINE.finditer(masked):
        parts = [int(x) for x in m.group(1).split(".")]
        if len(parts) == 1:
            if parts[0] == expect:
                tops.append((m.start(), parts[0])); cur = expect; expect += 1
        elif cur and parts[0] == cur:
            subs.append(m.start())
    if len(tops) < 2:                 # a lonely "1." is not an outline
        return [], subs if tops else []
    # numbered rows of a layout/table ("5. HORA N 4 80 M HHMM") are not topics:
    # real topics have some body text
    sizes = [b - a for (a, _), (b, _) in zip(tops, tops[1:] + [(len(text), 0)])]
    if sorted(sizes)[len(sizes) // 2] < 120:
        return [], []
    return tops, subs

def split_big(text):
    out = []
    for para in text.split("\n\n"):
        if len(para) <= MAX_CHARS: out.append(para); continue
        buf = ""
        for s in SENT.split(para):
            while len(s) > MAX_CHARS:
                out.append(s[:MAX_CHARS]); s = s[MAX_CHARS:]
            if len(buf) + len(s) + 1 > MAX_CHARS: out.append(buf); buf = ""
            buf = f"{buf} {s}".strip()
        if buf: out.append(buf)
    return out

def pack(pieces):
    chunks, buf = [], ""
    for p in pieces:
        if buf and len(buf) + len(p) + 2 > MAX_CHARS and len(buf) >= MIN_CHARS:
            chunks.append(buf); buf = ""
        buf = f"{buf}\n\n{p}".strip()
    if buf:
        if chunks and len(buf) < MIN_CHARS and len(chunks[-1]) + len(buf) <= MAX_CHARS * 1.25:
            chunks[-1] += "\n\n" + buf
        else:
            chunks.append(buf)
    return chunks

def split_outline(text, prefix, limit):
    """Split an oversized outline item by its direct children (1.1, 1.2 ...);
    a child that is still too big is split by its own children (1.1.1 ...),
    and only then by paragraphs/sentences. Keeps steps of a procedure together
    as long as possible."""
    if len(text) <= limit:
        return [text]
    kids = [m.start() for m in OUTLINE.finditer(text)
            if m.group(1).startswith(prefix + ".") and m.group(1).count(".") == prefix.count(".") + 1]
    kids = [k for k in kids if k > 0]
    if not kids:
        labels = [m.start() for m in LABEL.finditer(text) if m.start() > 0]
        parts = split_at(text, labels) if labels else [text]
        return [q for part in parts for q in (split_big(part) if len(part) > limit else [part])]
    out = []
    for part in split_at(text, kids):
        m = OUTLINE.match(part)
        out += split_outline(part, m.group(1), limit) if m else \
               (split_big(part) if len(part) > limit else [part])
    return out

def topic_title(text):
    first = re.split(r"\s\d{1,2}\.\d{1,2}[.)]\s|\n", text, maxsplit=1)[0]
    first = re.sub(r"^\d{1,2}\s?[.)=]\s*", "", first)
    return first[:140].strip()

# ---------------------------------------------------------------- main
def chunk_document(doc_id, name, text, cleaned=None):
    title = re.sub(r"\.(pdf|docx)$", "", name, flags=re.I)
    doc_programs = codes(name)
    text = cleaned if cleaned is not None else clean(text)
    records = []
    for block in SEP.split(text):
        for rec in split_on(REC_START, block):
            # a record that is only screenshots belongs to the record before it
            if records and len(substance(rec)) < 30:
                records[-1] += " " + rec
            elif records and len(substance(records[-1])) < 30:
                records[-1] = records[-1] + "\n\n" + rec
            else:
                records.append(rec)

    out, pos, topic_no = [], 0, 0
    topics_out = chunk_document.topics
    for r_i, rec in enumerate(records):
        ped = PEDIDO.search(rec)
        rtype = ("pedido" if ped else "email" if rec.startswith(("De:", "E-mail —"))
                 else "teste" if rec.lower().startswith("testes") else "documentacao")

        tops, subs = outline(rec)
        whole = bool(tops) and len(rec) <= TOPIC_MAX
        if whole:
            # a short numbered flow (steps 1..N) reads as one procedure: keep it whole
            topics = [("", rec)]
        elif tops:
            bounds = [p for p, _ in tops]
            intro = rec[:bounds[0]].strip()
            topics = ([("", intro)] if intro else []) + \
                     [(str(n), t) for (_, n), t in zip(tops, split_at(rec, bounds)[1 if bounds[0] > 0 else 0:])]
        else:
            topics = [("", rec)]

        prev_topic_id = None
        for num, ttext in topics:
            t_title = topic_title(ttext) if num else ""
            t_programs = codes(t_title) or doc_programs
            if whole or (num and len(ttext) <= TOPIC_MAX):
                pieces = [ttext]                       # whole procedure in one chunk
            elif num:
                pieces = split_outline(ttext, num, MAX_CHARS)
            else:
                pieces = []
                for sec in split_on(LABEL, ttext):
                    pieces += split_big(sec) if len(sec) > MAX_CHARS else [sec]
            topic_id = f"{doc_id}-t{topic_no}"; topic_no += 1
            prev_tail = ""
            chunks_here = pieces if ((num or whole) and len(pieces) == 1) else pack(pieces)
            if num:
                topics_out.append({"topic_id": topic_id, "doc_id": doc_id, "doc_name": name,
                                   "topic_number": num, "topic_title": t_title,
                                   "programs": " ".join(t_programs),
                                   "previous_topic_id": prev_topic_id, "text": ttext,
                                   "n_chunks": len(chunks_here)})
            for part_no, body in enumerate(chunks_here):
                steps = [m.group(1) for m in OUTLINE.finditer(body) if num and m.group(1).startswith(num + ".")]
                rng = "" if not steps else f"passo {steps[0]}" if steps[0] == steps[-1] \
                      else f"passos {steps[0]} a {steps[-1]}"
                span = f" (parte {part_no+1}/{len(chunks_here)}{', ' + rng if rng else ''})" \
                       if len(chunks_here) > 1 else ""
                header = (f"[{title}]" + (f" tópico {num}: {t_title}{span}" if num else "") +
                          f" | programa: {', '.join(t_programs) or '-'} | tipo: {rtype}" +
                          (f" | pedido: {ped.group(1)}" if ped else ""))
                full = body if not prev_tail else f"(...) {prev_tail}\n\n{body}"
                out.append({
                    "chunk_id": f"{doc_id}-{pos}", "doc_id": doc_id, "doc_name": name,
                    "position": pos, "record": r_i, "record_type": rtype,
                    "pedido": ped.group(1) if ped else "",
                    "topic_id": topic_id, "topic_number": num, "topic_title": t_title,
                    "previous_topic_id": prev_topic_id if num else None,   # "next step" link
                    "main_programs": " ".join(t_programs),
                    "mentioned_programs": " ".join(codes(body)),
                    "code_tables": " ".join(sorted({x for m in re.findall(r"\[Código: ([^\]]*)\]", body) for x in m.split()})),
                    "text": body,
                    "embed_text": f"{header}\n{full}",
                })
                prev_tail = body[-OVERLAP:].split(" ", 1)[-1]
                pos += 1
            if num: prev_topic_id = topic_id
    return absorb_image_only(out, doc_id)

MSG_CODE = re.compile(r"\b(?:KND|ORA)-\s?\d+")

def substance(text):
    """Letters/digits left once the image references are removed."""
    text = re.sub(r"\[(?:Imagem|Código):[^\]]*\]", "", text)
    text = re.sub(r"^E-mail —(?: Assunto:)?", "", text.strip())
    return re.sub(r"\W", "", text)

def absorb_image_only(chunks, doc_id):
    """A chunk that is only '[Imagem: EPRO15 | KND-000233]' is not worth storing
    on its own. Its program codes and messages move to the neighbouring chunk
    (previous one in the same record, otherwise the next one) and it is dropped."""
    for c in chunks:
        c["image_refs"] = []
        c["messages"] = sorted({re.sub(r"\s", "", m) for m in MSG_CODE.findall(c["text"])})
    keep = []
    pending = []                      # refs waiting for a next chunk
    for c in chunks:
        if len(substance(c["text"])) < 30:
            absorb_image_only.used += 1
            refs = re.findall(r"\[Imagem: ([^\]]*)\]", c["text"])
            target = keep[-1] if keep and keep[-1]["record"] == c["record"] else None
            if target is not None:
                attach(target, refs, c)
            else:
                pending.append((refs, c))
            continue
        for refs, old in pending:
            attach(c, refs, old)
        pending = []
        keep.append(c)
    if pending and keep:
        for refs, old in pending:
            attach(keep[-1], refs, old)
    for i, c in enumerate(keep):        # renumber
        c["position"] = i
        c["chunk_id"] = f"{doc_id}-{i}"
    return keep

absorb_image_only.used = 0

def attach(target, refs, dropped):
    for r in refs:
        if r not in target["image_refs"]:
            target["image_refs"].append(r)
    progs = set(target["mentioned_programs"].split()) | set(dropped["mentioned_programs"].split())
    target["mentioned_programs"] = " ".join(sorted(progs))
    target["messages"] = sorted(set(target["messages"]) | set(dropped["messages"]))
    target["code_tables"] = " ".join(sorted(set(target["code_tables"].split()) | set(dropped["code_tables"].split())))
    # keep a short leftover (e.g. "4 3º passo") only if it is not just an image
    left = re.sub(r"\[Imagem:[^\]]*\]", "", dropped["text"]).strip()
    if left and len(left) > 3:
        target["text"] += "\n\n" + left
        target["embed_text"] += "\n\n" + left

chunk_document.topics = []

def shingles(text):
    w = re.findall(r"\w+", re.sub(r"\[(?:Imagem|Código):[^\]]*\]", " ", text.lower()))
    return {" ".join(w[i:i + 5]) for i in range(len(w) - 4)}

def find_copies(cleaned, names, min_overlap=0.6, min_size_ratio=0.7):
    """Documents that are copies or older versions of another document.
    A is dropped when >= 60% of its 5-word sequences are also in B, B is at least
    as long, and both have a similar size (so a short manual that is *included*
    in a big compilation is kept). Returns {dropped_id: kept_id}."""
    from collections import Counter, defaultdict
    S = [shingles(t) for t in cleaned]
    df = Counter(x for s in S for x in s)
    inv = defaultdict(list)
    for i, s in enumerate(S):
        for x in s:
            if df[x] <= 20: inv[x].append(i)       # ignore boilerplate shared by many docs
    dropped = {}
    order = sorted(range(len(S)), key=lambda i: (len(S[i]), names[i]))   # smallest first
    for i in order:
        if len(S[i]) < 30 or i in dropped: continue
        cnt = Counter(j for x in S[i] if df[x] <= 20 for j in inv[x] if j != i and j not in dropped)
        best = None
        for j, n in cnt.items():
            ci, cj = set(codes(names[i])), set(codes(names[j]))
            if ci and cj and ci != cj:
                continue       # same template, different programs (e.g. CFAB109 vs CFAB110)
            if n / len(S[i]) >= min_overlap and len(S[j]) >= len(S[i]) \
                    and len(S[i]) / len(S[j]) >= min_size_ratio:
                if best is None or n > best[1]: best = (j, n)
        if best:
            dropped[i] = best[0]
    for i in dropped:                          # A -> B -> C : point A to the document kept
        while dropped[i] in dropped: dropped[i] = dropped[dropped[i]]
    return dropped

if __name__ == "__main__":
    from collections import Counter
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    keep_copies = "--keep-duplicates" in sys.argv
    docs = json.load(open(args[0] if args else "export.json", encoding="utf-8"))
    cleaned = [clean(d["text"]) for d in docs]
    copies = {} if keep_copies else find_copies(cleaned, [d["name"] for d in docs])
    with open("duplicates.csv", "w", encoding="utf-8") as f:
        f.write("dropped_doc_id;dropped_doc;kept_doc_id;kept_doc\n")
        for i, j in sorted(copies.items()):
            f.write(f"{i};{docs[i]['name']};{j};{docs[j]['name']}\n")
    rows = [c for i, d in enumerate(docs) if i not in copies
            for c in chunk_document(i, d["name"], d["text"], cleaned[i])]
    print(f"documents skipped as copies/older versions: {len(copies)} (see duplicates.csv; "
          f"--keep-duplicates to keep them)")
    per_topic = Counter(r["topic_id"] for r in rows)
    prev_of = {}
    for t in chunk_document.topics:
        t["n_chunks"] = per_topic.get(t["topic_id"], 0)
        prev_of[t["topic_id"]] = t["previous_topic_id"]
    # topics that were only images are dropped; links skip over them
    empty = {t["topic_id"] for t in chunk_document.topics if not t["n_chunks"]}
    def real_prev(tid):
        while tid in empty: tid = prev_of[tid]
        return tid
    chunk_document.topics = [t for t in chunk_document.topics if t["n_chunks"]]
    for t in chunk_document.topics: t["previous_topic_id"] = real_prev(t["previous_topic_id"])
    for r in rows:
        if r["previous_topic_id"]: r["previous_topic_id"] = real_prev(r["previous_topic_id"])
    with open("chunks.jsonl", "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open("topics.jsonl", "w", encoding="utf-8") as f:
        for t in chunk_document.topics: f.write(json.dumps(t, ensure_ascii=False) + "\n")
    import statistics as st
    L = [len(r["text"]) for r in rows]
    print(f"{len(docs)} docs -> {len(rows)} chunks | chars median {st.median(L):.0f} "
          f"p95 {sorted(L)[int(len(L)*.95)]} max {max(L)} | <{MIN_CHARS}: {sum(l<MIN_CHARS for l in L)}")
    print(Counter(r["record_type"] for r in rows))
    print("chunks inside a numbered topic:", sum(1 for r in rows if r["topic_number"]),
          "| docs with numbered topics:", len({r["doc_id"] for r in rows if r["topic_number"]}),
          "| topics kept whole:", sum(t["n_chunks"] == 1 for t in chunk_document.topics),
          "of", len(chunk_document.topics))
    print("image-only chunks absorbed by the safety net:", absorb_image_only.used)
