"""
경기 요약 리포트: 분석 결과를 csv 3개(랠리/선수/타격)로 저장하고, 그 csv만 읽어 그림이 들어간 html 한 장으로 만든다.
    python report.py outputs/분석_x          # 분석_x_랠리.csv, _선수.csv, _타격.csv -> 분석_x_리포트.html
앱/CLI는 save_all()을 부르면 csv와 html을 함께 만든다.
득점 '방식'(위너/에러/샷 종류)은 공 모델과 샷 분류가 생긴 뒤에 붙일 수 있다. 지금 득점자는 '마지막 타격자' 추정값이다.
"""
import csv
import html
import sys

import rally

COLORS = ["#38d982", "#ffb224"]  # overlay_ui.PLAYER_COLORS와 같은 초록/주황
GREY = "#8a94a6"


def _write(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def save_csvs(data, names, stem):
    """data(postprocess를 거친 분석 결과) -> stem_랠리/선수/타격.csv. 경로 3개를 돌려준다."""
    fps, start, frames = data["fps"], float(data.get("start") or 0), data["frames"]
    n = max((len(f["p"]) for f in frames), default=0)
    name = lambda k: names[k] if k < len(names) and str(names[k]).strip() else f"P{k + 1}"
    dist_at = lambda i, k: frames[min(max(i, 0), len(frames) - 1)]["p"][k]["dist"] if k < len(frames[min(max(i, 0), len(frames) - 1)]["p"]) else 0.0

    rows = rally.export_rows(data["rallies"], fps, start, [name(k) for k in range(max(n, 2))])
    for row, r in zip(rows, data["rallies"]):
        row["reason"] = r.get("reason", "")
        for k in range(n):
            row[f"shots_{name(k)}"] = sum(h["slot"] == k for h in r["hits"])
            row[f"dist_m_{name(k)}"] = round(dist_at(r["end"], k) - dist_at(r["start"], k), 1)
    fields = list(rows[0]) if rows else ["rally", "start_s", "end_s", "duration_s", "n_shots", "winner", "manual", "reason"]
    p_rally = f"{stem}_랠리.csv"
    _write(p_rally, rows, fields)

    points = rally.score_at(data["rallies"], 10 ** 9, fps) if data["rallies"] else [0] * n
    players = []
    for k in range(n):
        vis = [f["p"][k] for f in frames if len(f["p"]) > k and f["p"][k]["v"]]
        players.append(dict(
            name=name(k), slot=k + 1, dist_m=round(frames[-1]["p"][k]["dist"], 1) if frames else 0,
            dist_estimated=int(not data["calibrated"]), shots=sum(h["slot"] == k for h in data["hits"]),
            points=points[k], avg_speed_mps=round(sum(p["speed"] for p in vis) / max(len(vis), 1), 2),
            tracked_pct=round(100 * len(vis) / max(len(frames), 1))))
    p_players = f"{stem}_선수.csv"
    _write(p_players, players, ["name", "slot", "dist_m", "dist_estimated", "shots", "points", "avg_speed_mps", "tracked_pct"])

    owner = {}
    for r in data["rallies"]:
        for h in r["hits"]:
            owner[h["frame"]] = r["id"]
    shots = [dict(no=i + 1, time_s=round(start + h["frame"] / fps, 1), rally=owner.get(h["frame"], ""),
                  player=name(h["slot"]), source=h.get("source", "")) for i, h in enumerate(data["hits"])]
    p_shots = f"{stem}_타격.csv"
    _write(p_shots, shots, ["no", "time_s", "rally", "player", "source"])
    return p_rally, p_players, p_shots


def _read(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _bars(rows, key, unit, label):
    """선수별 가로 막대 한 묶음."""
    top = max([float(r[key]) for r in rows] + [1e-9])
    out = [f'<div class="bt">{label}</div>']
    for i, r in enumerate(rows):
        v = float(r[key])
        out.append(f'<div class="br"><span class="bn">{html.escape(r["name"])}</span>'
                   f'<span class="bb"><i style="width:{100 * v / top:.1f}%;background:{COLORS[i % 2]}"></i></span>'
                   f'<b>{v:g}{unit}</b></div>')
    return "".join(out)


def _score_chart(rallies, names):
    """랠리가 끝날 때마다 점수가 어떻게 변했는지 계단선."""
    w, h, pad = 640, 220, 34
    sc, pts = [0, 0], [(0, 0, 0)]
    for r in rallies:
        if r["winner"] in names:
            sc[names.index(r["winner"])] += 1
        pts.append((int(r["rally"]), sc[0], sc[1]))
    top = max(max(sc), 1)
    x = lambda i: pad + (w - 2 * pad) * i / max(len(pts) - 1, 1)
    y = lambda v: h - pad - (h - 2 * pad) * v / top
    svg = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="점수 변화">']
    for v in range(top + 1):
        svg.append(f'<line x1="{pad}" x2="{w - pad}" y1="{y(v)}" y2="{y(v)}" stroke="#2a3140"/>'
                   f'<text x="{pad - 8}" y="{y(v) + 4}" text-anchor="end" fill="{GREY}" font-size="11">{v}</text>')
    for k in range(2):
        d = f"M{x(0)},{y(pts[0][1 + k])}"
        for i in range(1, len(pts)):
            d += f" H{x(i)} V{y(pts[i][1 + k])}"
        svg.append(f'<path d="{d}" fill="none" stroke="{COLORS[k]}" stroke-width="3" stroke-linejoin="round"/>')
    for i in range(1, len(pts)):
        svg.append(f'<text x="{x(i)}" y="{h - pad + 16}" text-anchor="middle" fill="{GREY}" font-size="10">{pts[i][0]}</text>')
    svg.append(f'<text x="{w / 2}" y="{h - 4}" text-anchor="middle" fill="{GREY}" font-size="11">랠리 번호</text></svg>')
    return "".join(svg)


def _rally_bars(rallies, names):
    """랠리별 타격 수 막대(색 = 득점자, 회색 = 무효/미정)."""
    w, h, pad = 640, 200, 34
    top = max([int(r["n_shots"]) for r in rallies] + [1])
    bw = (w - 2 * pad) / max(len(rallies), 1)
    svg = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="랠리별 타격 수">']
    for i, r in enumerate(rallies):
        n = int(r["n_shots"])
        col = COLORS[names.index(r["winner"]) % 2] if r["winner"] in names else GREY
        bh = (h - 2 * pad) * n / top
        bx = pad + i * bw + bw * 0.15
        svg.append(f'<rect x="{bx:.1f}" y="{h - pad - bh:.1f}" width="{bw * 0.7:.1f}" height="{bh:.1f}" rx="3" fill="{col}">'
                   f'<title>랠리 {r["rally"]}: 타격 {n}회, {r["duration_s"]}초, 득점 {html.escape(r["winner"] or "-")}</title></rect>'
                   f'<text x="{bx + bw * 0.35:.1f}" y="{h - pad - bh - 4:.1f}" text-anchor="middle" fill="#cfd6e4" font-size="11">{n}</text>'
                   f'<text x="{bx + bw * 0.35:.1f}" y="{h - pad + 16}" text-anchor="middle" fill="{GREY}" font-size="10">{r["rally"]}</text>')
    svg.append(f'<text x="{w / 2}" y="{h - 4}" text-anchor="middle" fill="{GREY}" font-size="11">랠리 번호 · 막대 색 = 득점한 선수</text></svg>')
    return "".join(svg)


def html_from_csvs(p_rally, p_players, p_shots, title="경기 요약 리포트"):
    """csv 3개만 읽어 html 문자열을 만든다."""
    rallies, players, shots = _read(p_rally), _read(p_players), _read(p_shots)
    names = [p["name"] for p in players]
    est = any(p["dist_estimated"] == "1" for p in players)
    durs = [float(r["duration_s"]) for r in rallies]
    longest = max(rallies, key=lambda r: int(r["n_shots"]), default=None)
    final = " : ".join(p["points"] for p in players)
    legend = "".join(f'<span><i style="background:{COLORS[i % 2]}"></i>{html.escape(n)}</span>' for i, n in enumerate(names))
    cards = [("최종 점수", final), ("랠리 수", len(rallies)), ("평균 랠리", f"{sum(durs) / max(len(durs), 1):.1f}초"),
             ("가장 긴 랠리", f'타격 {longest["n_shots"]}회 (#{longest["rally"]})' if longest else "-"),
             ("전체 타격", len(shots))]
    dcols = [c for c in (rallies[0] if rallies else {}) if c.startswith("dist_m_")]
    scols = [c for c in (rallies[0] if rallies else {}) if c.startswith("shots_")]
    head = "".join(f"<th>{h}</th>" for h in ["#", "시작", "길이", "타격"] + [c[6:] + " 타격" for c in scols] +
                   [c[7:] + " 이동(m)" for c in dcols] + ["득점", "근거"])
    body = ""
    for r in rallies:
        cells = [r["rally"], f'{r["start_s"]}s', f'{r["duration_s"]}s', r["n_shots"]] + [r[c] for c in scols] + \
                [r[c] for c in dcols] + [(r["winner"] or "무효") + (" ✎" if r["manual"] == "True" else ""), r["reason"]]
        body += "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in cells) + "</tr>"
    notes = ["득점자는 '마지막에 친 사람'으로 추정한 값입니다(✎ = 사람이 고친 값). 틴/아웃 판정과 샷 종류별 득점 방식은 공 모델이 생기면 추가됩니다.",
             "타격 횟수는 손목 움직임(공 모델이 있으면 공 궤적)으로 센 추정값입니다."]
    if est:
        notes.append("이동 거리는 코트 보정 없이 화면 크기로 환산한 대략값(≈)입니다. 코트 점을 찍으면 정확해집니다.")
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>
body{{margin:0;background:#0f141c;color:#e6ebf5;font:15px/1.5 -apple-system,"Apple SD Gothic Neo","Malgun Gothic",sans-serif}}
main{{max-width:760px;margin:0 auto;padding:24px 16px 48px}}h1{{font-size:22px;margin:0 0 4px}}h2{{font-size:16px;margin:28px 0 10px}}
.lg span{{margin-right:14px}}.lg i,.bb i{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-top:16px}}
.card{{background:#171d28;border-radius:12px;padding:12px 14px}}.card small{{color:{GREY};display:block}}.card b{{font-size:20px}}
.box{{background:#171d28;border-radius:12px;padding:14px 16px}}svg{{width:100%;height:auto;display:block}}
.bt{{color:{GREY};font-size:13px;margin:10px 0 4px}}.br{{display:flex;align-items:center;gap:10px;margin:4px 0}}
.bn{{width:70px}}.bb{{flex:1;background:#232b3a;border-radius:6px;height:10px;overflow:hidden}}.bb i{{display:block;height:10px;border-radius:6px;margin:0}}
.br b{{width:72px;text-align:right}}.tw{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{padding:6px 8px;border-bottom:1px solid #232b3a;text-align:left;white-space:nowrap}}th{{color:{GREY};font-weight:600}}
.note{{color:{GREY};font-size:12.5px;margin:6px 0}}
</style></head><body><main>
<h1>{html.escape(title)}</h1><div class="lg">{legend}</div>
<div class="cards">{"".join(f'<div class="card"><small>{a}</small><b>{b}</b></div>' for a, b in cards)}</div>
<h2>선수 비교</h2><div class="box">{_bars(players, "dist_m", " m", "이동 거리" + (" (대략)" if est else ""))}{_bars(players, "shots", "회", "타격")}{_bars(players, "points", "점", "득점")}{_bars(players, "avg_speed_mps", " m/s", "평균 이동 속도")}
<div class="bt">화면에서 추적된 비율: {" · ".join(f'{html.escape(p["name"])} {p["tracked_pct"]}%' for p in players)}</div></div>
<h2>점수 변화</h2><div class="box">{_score_chart(rallies, names)}</div>
<h2>랠리별 타격 수</h2><div class="box">{_rally_bars(rallies, names)}</div>
<h2>랠리 기록</h2><div class="box tw"><table><tr>{head}</tr>{body}</table></div>
{"".join(f'<p class="note">· {html.escape(n)}</p>' for n in notes)}
</main></body></html>"""


def save_all(data, names, stem, title="경기 요약 리포트"):
    """csv 3개 + html 리포트를 만들고 html 경로를 돌려준다."""
    p_rally, p_players, p_shots = save_csvs(data, names, stem)
    out = f"{stem}_리포트.html"
    open(out, "w", encoding="utf-8").write(html_from_csvs(p_rally, p_players, p_shots, title))
    return out


def _self_check():
    import os
    import tempfile
    hits = [dict(frame=f, slot=s, source="swing") for f, s in ((10, 0), (40, 1), (70, 0), (400, 1), (430, 0))]
    data = dict(fps=30, start=0.0, calibrated=False, hits=hits,
                frames=[dict(p=[dict(v=1, dist=i * 0.01, speed=1.0), dict(v=1, dist=i * 0.02, speed=2.0)]) for i in range(500)])
    data["rallies"] = rally.apply_winners(rally.split_rallies(hits, 30), {2: None})
    with tempfile.TemporaryDirectory() as d:
        out = save_all(data, ["민수", "지훈"], os.path.join(d, "t"))
        page = open(out, encoding="utf-8").read()
    assert "민수" in page and "<svg" in page and "무효" in page and "대략값" in page
    print("self-check OK")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _self_check()
    else:
        s = sys.argv[1]
        out = f"{s}_리포트.html"
        open(out, "w", encoding="utf-8").write(html_from_csvs(f"{s}_랠리.csv", f"{s}_선수.csv", f"{s}_타격.csv"))
        print("saved", out)
