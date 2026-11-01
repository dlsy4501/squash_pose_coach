"""
타격 목록을 랠리로 나누고, 랠리마다 득점자를 정해 점수표를 만든다 (스쿼시 PAR: 랠리마다 1점).

득점자 추정은 아직 단순한 기본값이다: 랠리의 마지막 타격자를 득점자로 본다(상대가 못 받았거나
공이 좋았다고 가정). 틴/아웃 같은 실수 판정은 공 궤적이 안정적으로 나와야 가능하고, 그때까지는
사람이 앱에서 랠리별로 바로잡는다(overrides). 바로잡은 값은 나중에 추정 규칙을 평가하는 정답으로 쓸 수 있다.
"""
RALLY_GAP_S = 3.0     # 이 시간 이상 타격이 없으면 랠리가 끝난 것
DECIDE_DELAY_S = 1.0  # 마지막 타격 후 이만큼 지나면 점수에 반영해서 보여준다


def split_rallies(hits, fps):
    """hits: [{frame, slot, ...}] (프레임 순). 반환: [{id, start, end, hits, guess, reason}]"""
    rallies = []
    for h in hits:
        if rallies and h["frame"] - rallies[-1]["hits"][-1]["frame"] <= RALLY_GAP_S * fps:
            rallies[-1]["hits"].append(h)
        else:
            rallies.append(dict(id=len(rallies) + 1, hits=[h]))
    for r in rallies:
        r["start"], r["end"] = r["hits"][0]["frame"], r["hits"][-1]["frame"]
        r["guess"] = r["hits"][-1]["slot"]
        r["reason"] = "마지막 타격자 (추정)"
    return rallies


def apply_winners(rallies, overrides=None):
    """overrides: {랠리 id: 슬롯 또는 None(무효 랠리, 점수 없음)}. 각 랠리에 winner와 decided(점수 반영 프레임 기준)를 채운다."""
    overrides = overrides or {}
    for r in rallies:
        r["winner"] = overrides.get(r["id"], r["guess"])
        r["manual"] = r["id"] in overrides
    return rallies


def score_at(rallies, frame, fps, n_players=2):
    """frame 시점에 화면에 보여줄 점수."""
    score = [0] * n_players
    for r in rallies:
        if r["winner"] is not None and r["end"] + DECIDE_DELAY_S * fps <= frame:
            score[r["winner"]] += 1
    return score


def to_rows(rallies, fps, start=0.0):
    """앱 표시용: [랠리, 시작(초), 타격 수, 추정 득점자, 근거]"""
    return [[r["id"], round(start + r["start"] / fps, 1), len(r["hits"]), r["guess"], r["reason"]] for r in rallies]


def export_rows(rallies, fps, start=0.0, names=None):
    """랠리 기록 저장용: 랠리마다 시작/끝/길이(초), 타격 수, 득점자(이름), 정정 여부.
    names가 있으면 득점자를 이름으로, 없으면 슬롯 번호로 적는다. 무효 랠리는 득점자 ""."""
    who = (lambda k: names[k]) if names else (lambda k: k)
    rows = []
    for r in rallies:
        s0, s1 = start + r["start"] / fps, start + r["end"] / fps
        rows.append(dict(rally=r["id"], start_s=round(s0, 1), end_s=round(s1, 1), duration_s=round(s1 - s0, 1),
                         n_shots=len(r["hits"]), winner=who(r["winner"]) if r["winner"] is not None else "",
                         manual=r.get("manual", False)))
    return rows


def save_csv(rallies, fps, path, start=0.0, names=None):
    """export_rows 결과를 csv로 저장한다."""
    import csv
    rows = export_rows(rallies, fps, start, names)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["rally", "start_s", "end_s", "duration_s", "n_shots", "winner", "manual"])
        w.writeheader()
        w.writerows(rows)
    return path


def _self_check():
    H = lambda f, s: dict(frame=f, slot=s)
    # 랠리 1: 0,1,0 번 타격 (마지막 0) / 3초 넘게 쉬고 랠리 2: 1,0,1 (마지막 1) / 랠리 3: 타격 1번
    hits = [H(0, 0), H(30, 1), H(60, 0), H(300, 1), H(330, 0), H(360, 1), H(900, 0)]
    rs = split_rallies(hits, 30)
    assert [len(r["hits"]) for r in rs] == [3, 3, 1], rs
    assert [r["guess"] for r in rs] == [0, 1, 0]

    apply_winners(rs, {2: 0, 3: None})  # 사용자가 2번 랠리를 0번 선수 득점으로 정정, 3번은 무효
    assert [r["winner"] for r in rs] == [0, 0, None] and rs[1]["manual"] and not rs[0]["manual"]

    assert score_at(rs, 0, 30) == [0, 0]
    assert score_at(rs, 60 + 29, 30) == [0, 0]       # 랠리 끝난 직후엔 아직 반영 전
    assert score_at(rs, 60 + 30, 30) == [1, 0]
    assert score_at(rs, 10**6, 30) == [2, 0]         # 무효 랠리는 점수에 안 들어감
    assert to_rows(rs, 30)[0] == [1, 0.0, 3, 0, "마지막 타격자 (추정)"]

    rows = export_rows(rs, 30, names=["민수", "지훈"])
    assert rows[0] == dict(rally=1, start_s=0.0, end_s=2.0, duration_s=2.0, n_shots=3, winner="민수", manual=False)
    assert rows[2]["winner"] == ""  # 무효 랠리는 득점자 없음
    import tempfile
    path = save_csv(rs, 30, tempfile.mktemp(suffix=".csv"), names=["민수", "지훈"])
    assert open(path, encoding="utf-8-sig").readline().strip() == "rally,start_s,end_s,duration_s,n_shots,winner,manual"
    print("self-check OK")


if __name__ == "__main__":
    _self_check()
