#!/usr/bin/env python3
"""Claude Code kullanım paneli: ~/.claude/projects altındaki kayıtlardan
günlük / saatlik token kullanımını okuyup tek dosyalık HTML üretir.

Kullanım: python3 usage_panel.py [--dir ~/.claude/projects] [--out usage.html]
"""
import argparse, json, glob, os, html
from datetime import datetime

FIELDS = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")


def load(root):
    seen, rows = set(), []
    for path in glob.glob(os.path.join(root, "**", "*.jsonl"), recursive=True):
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if '"usage"' not in line:
                    continue
                try:
                    d = json.loads(line)
                    m = d.get("message") or {}
                    u = m.get("usage")
                    ts = d.get("timestamp")
                    if not u or not ts:
                        continue
                    key = (m.get("id"), d.get("requestId"))
                    if key[0] and key in seen:
                        continue
                    seen.add(key)
                    t = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone()
                    rows.append((t, m.get("model", "?"), [int(u.get(k) or 0) for k in FIELDS],
                                 os.path.basename(os.path.dirname(path))))
                except Exception:
                    continue
    return rows


def agg(rows):
    days, hours, models, projects = {}, [[0] * 4 for _ in range(24)], {}, {}
    for t, model, v, proj in rows:
        d = days.setdefault(t.strftime("%Y-%m-%d"), [0, 0, 0, 0, 0])
        for i in range(4):
            d[i] += v[i]
            hours[t.hour][i] += v[i]
        d[4] += 1
        models[model] = models.get(model, 0) + v[0] + v[1] + v[2] + v[3]
        projects[proj] = projects.get(proj, 0) + v[0] + v[1] + v[2] + v[3]
    return days, hours, models, projects


def fmt(n):
    return f"{n/1e6:.2f}M" if n >= 1e6 else f"{n/1e3:.1f}K" if n >= 1e3 else str(n)


def bars(items, total_fn, label_fn):
    mx = max([total_fn(v) for _, v in items] + [1])
    out = []
    for k, v in items:
        tot = total_fn(v)
        out.append(f'<div class="row"><span class="l">{html.escape(label_fn(k))}</span>'
                   f'<span class="b"><i style="width:{tot / mx * 100:.1f}%"></i></span>'
                   f'<span class="v">{fmt(tot)}</span></div>')
    return "\n".join(out)


def render(rows):
    days, hours, models, projects = agg(rows)
    allt = sum(sum(v[:4]) for v in days.values())
    day_items = sorted(days.items(), reverse=True)
    day_rows = "\n".join(
        f"<tr><td>{k}</td><td>{v[4]}</td><td>{fmt(v[0])}</td><td>{fmt(v[1])}</td>"
        f"<td>{fmt(v[2])}</td><td>{fmt(v[3])}</td><td><b>{fmt(sum(v[:4]))}</b></td></tr>"
        for k, v in day_items)
    hour_bars = bars([(h, v) for h, v in enumerate(hours)], sum, lambda h: f"{h:02d}:00")
    day_bars = bars(sorted(days.items())[-30:], lambda v: sum(v[:4]), lambda k: k[5:])
    model_bars = bars(sorted(models.items(), key=lambda x: -x[1]), lambda v: v, lambda k: k)
    proj_bars = bars(sorted(projects.items(), key=lambda x: -x[1])[:10], lambda v: v, lambda k: k.lstrip("-"))
    return f"""<!doctype html><html lang="tr"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Claude Kullanım Paneli</title>
<style>
:root{{--bg:#fafaf8;--fg:#1f1e1d;--mut:#76736c;--card:#fff;--bd:#e5e3dc;--ac:#c96442}}
@media(prefers-color-scheme:dark){{:root{{--bg:#1c1b19;--fg:#ece9e1;--mut:#9a968c;--card:#262522;--bd:#38362f;--ac:#e08a69}}}}
body{{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif}}
main{{max-width:900px;margin:auto}}h1{{font-size:20px}}h2{{font-size:15px;margin:0 0 10px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:14px}}
.card,section{{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:12px;margin-bottom:14px}}
.cards .card{{margin:0}}.card small{{color:var(--mut)}}.card div{{font-size:22px;font-weight:600}}
.row{{display:flex;align-items:center;gap:8px;margin:3px 0;font-size:12px}}
.l{{width:90px;color:var(--mut);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.b{{flex:1;background:var(--bd);height:10px;border-radius:5px;overflow:hidden}}
.b i{{display:block;height:100%;background:var(--ac)}}.v{{width:60px;text-align:right}}
table{{width:100%;border-collapse:collapse;font-size:12px}}th,td{{text-align:right;padding:5px;border-bottom:1px solid var(--bd)}}
th:first-child,td:first-child{{text-align:left}}.sc{{overflow-x:auto}}
</style><main>
<h1>Claude Kullanım Paneli</h1>
<div class="cards">
<div class="card"><small>Toplam token</small><div>{fmt(allt)}</div></div>
<div class="card"><small>Gün sayısı</small><div>{len(days)}</div></div>
<div class="card"><small>İstek sayısı</small><div>{sum(v[4] for v in days.values())}</div></div>
<div class="card"><small>Çıktı tokenı</small><div>{fmt(sum(v[1] for v in days.values()))}</div></div></div>
<section><h2>Günlük kullanım (son 30 gün)</h2>{day_bars}</section>
<section><h2>Saate göre kullanım (yerel saat, tüm günler)</h2>{hour_bars}</section>
<section><h2>Model</h2>{model_bars}</section>
<section><h2>Proje</h2>{proj_bars}</section>
<section class="sc"><h2>Gün detayı</h2><table><tr><th>Gün</th><th>İstek</th><th>Girdi</th><th>Çıktı</th><th>Cache yazma</th><th>Cache okuma</th><th>Toplam</th></tr>
{day_rows}</table></section>
<small style="color:var(--mut)">Oluşturulma: {datetime.now():%Y-%m-%d %H:%M}</small></main></html>"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.expanduser("~/.claude/projects"))
    ap.add_argument("--out", default="usage.html")
    a = ap.parse_args()
    rows = load(a.dir)
    open(a.out, "w", encoding="utf-8").write(render(rows))
    print(f"{len(rows)} istek okundu -> {a.out}")
