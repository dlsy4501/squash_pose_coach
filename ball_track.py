"""
공 탐지 후처리. 모델이 프레임마다 내놓는 후보들을 궤적으로 정리하고, 방향이 급변하는
지점(타격/벽 반사/바닥 바운스)을 찾아 가장 가까운 선수 손목에 연결한다.

단계: pick_track(연속성으로 후보 고르기) -> remove_spikes(튀는 점 제거) -> interpolate(짧은 구멍 보간)
      -> find_turns(방향 급변) -> assign_hits(타격자 연결) -> merge_hits(손목 스윙과 합치기)

픽셀 임계값은 1080p 기준이고 영상 높이에 비례해 스케일한다(scale = 높이/1080).
ponytail: 임계값은 전부 경험치. 실제 공 모델 결과로 튜닝 필요.
"""
import math

MIN_CONF = 0.25
MAX_SPEED = 260     # px/frame: 한 프레임에 이보다 멀리 가면 같은 공으로 보지 않는다
RESET_GAP = 10      # 이 프레임 이상 못 보면 이전 위치 제약을 푼다
OUT_PX = 120        # 직선 보간값에서 이만큼 벗어나면 튀는 점 후보
INTERP_GAP = 5      # 이 프레임 이하의 빈 구간만 보간
TURN_DEG = 50       # 이 각도 이상 꺾이면 방향 급변
TURN_MIN_SPEED = 6  # px/frame: 이보다 느리면 방향이 의미 없다
HIT_RADIUS = 160    # px: 방향 급변 지점이 손목에서 이 안이면 그 선수의 타격
CONFIRM_WIN = 6     # 스윙 타격을 공 정보로 검증할 때 보는 앞뒤 프레임 수


def pick_track(cands_per_frame, scale=1.0):
    """프레임별 후보 [(x, y, conf), ...] -> 프레임별 한 점 또는 None.
    직전 공 위치에서 한 프레임에 갈 수 있는 거리 안의 후보만 인정하고 그중 conf가 최대인 것을 고른다."""
    track, prev, prev_i = [None] * len(cands_per_frame), None, None
    for i, cands in enumerate(cands_per_frame):
        cands = [c for c in cands if c[2] >= MIN_CONF]
        if prev is not None and i - prev_i <= RESET_GAP:
            lim = MAX_SPEED * scale * (i - prev_i)
            cands = [c for c in cands if math.dist(c[:2], prev[:2]) <= lim]
        if cands:
            track[i] = prev = max(cands, key=lambda c: c[2])
            prev_i = i
    return track


def _neighbors(track, i, span):
    j = next((k for k in range(i - 1, max(-1, i - span - 1), -1) if track[k]), None)
    k = next((k for k in range(i + 1, min(len(track), i + span + 1)) if track[k]), None)
    return j, k


def remove_spikes(track, scale=1.0, span=4):
    """앞뒤 점을 이은 직선에서 크게 벗어난 한 점(오검출)을 지운다. 급격한 반사점은 앞뒤 간격보다
    덜 벗어나므로 남는다."""
    drop = []
    for i, p in enumerate(track):
        if not p:
            continue
        j, k = _neighbors(track, i, span)
        if j is None or k is None:
            continue
        t = (i - j) / (k - j)
        ex = track[j][0] + (track[k][0] - track[j][0]) * t
        ey = track[j][1] + (track[k][1] - track[j][1]) * t
        resid = math.dist(p[:2], (ex, ey))
        if resid > max(OUT_PX * scale, 1.5 * math.dist(track[j][:2], track[k][:2])):
            drop.append(i)
    out = list(track)
    for i in drop:
        out[i] = None
    return out


def interpolate(track):
    """INTERP_GAP 이하의 빈 구간을 직선 보간. 결과: dict(x, y, conf, interp) 또는 None."""
    out = [dict(x=p[0], y=p[1], conf=p[2], interp=False) if p else None for p in track]
    last = None
    for i, p in enumerate(out):
        if p is None:
            continue
        if last is not None and 1 < i - last <= INTERP_GAP + 1:
            a = out[last]
            for k in range(last + 1, i):
                t = (k - last) / (i - last)
                out[k] = dict(x=a["x"] + (p["x"] - a["x"]) * t, y=a["y"] + (p["y"] - a["y"]) * t,
                              conf=min(a["conf"], p["conf"]), interp=True)
        last = i
    return out


def _angle(v1, v2):
    m = math.hypot(*v1) * math.hypot(*v2)
    if m == 0:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / m))))


def find_turns(points, scale=1.0):
    """공이 방향을 급히 꺾는 프레임들. kind: 'bounce'(화면에서 아래로 내려가다 위로 = 바닥 바운스 후보) / 'turn'."""
    cand = []
    for i in range(2, len(points) - 2):
        a, b, c = points[i - 2], points[i], points[i + 2]
        if not (a and b and c):
            continue
        v1 = ((b["x"] - a["x"]) / 2, (b["y"] - a["y"]) / 2)
        v2 = ((c["x"] - b["x"]) / 2, (c["y"] - b["y"]) / 2)
        if min(math.hypot(*v1), math.hypot(*v2)) < TURN_MIN_SPEED * scale:
            continue
        ang = _angle(v1, v2)
        if ang >= TURN_DEG:
            cand.append(dict(frame=i, x=b["x"], y=b["y"], angle=ang,
                             kind="bounce" if v1[1] > 0 > v2[1] else "turn"))
    turns = []  # 연속한 후보는 각도가 가장 큰 것만 남긴다
    for t in cand:
        if turns and t["frame"] - turns[-1]["frame"] <= 3:
            if t["angle"] > turns[-1]["angle"]:
                turns[-1] = t
        else:
            turns.append(t)
    return turns


def assign_hits(turns, wrists_at, scale=1.0):
    """wrists_at(frame) -> {slot: [(x, y), ...]}. 방향 급변 지점이 손목에서 HIT_RADIUS 안이면 그 선수의 타격."""
    hits = []
    for t in turns:
        best = None
        for slot, pts in wrists_at(t["frame"]).items():
            d = min((math.dist((t["x"], t["y"]), p) for p in pts), default=None)
            if d is not None and d <= HIT_RADIUS * scale and (best is None or d < best[1]):
                best = (slot, d)
        if best:
            hits.append(dict(frame=t["frame"], slot=best[0], source="ball"))
    return hits


def merge_hits(swing_hits, ball_hits, points, fps, min_gap_s=0.3):
    """손목 스윙 타격과 공 기반 타격을 합친다.
    - 스윙 타격은 '공이 앞뒤로 계속 보이는데 근처에서 방향이 안 꺾였다'면 연습 스윙으로 보고 버린다.
    - 공이 안 보이던 구간의 스윙 타격은 그대로 둔다.
    - 공 기반 타격은 근처 스윙 타격과 겹치면 하나로 합친다. 결과: 프레임 순 [{frame, slot, source}]"""
    out = []
    for s in swing_hits:
        lo, hi = max(0, s["frame"] - CONFIRM_WIN), min(len(points), s["frame"] + CONFIRM_WIN + 1)
        seen = sum(1 for p in points[lo:hi] if p)
        near_ball = any(abs(b["frame"] - s["frame"]) <= CONFIRM_WIN and b["slot"] == s["slot"] for b in ball_hits)
        if seen >= (hi - lo) * 0.7 and not near_ball:
            continue
        out.append(dict(s, source="swing+ball" if near_ball else "swing"))
    for b in ball_hits:
        if not any(abs(o["frame"] - b["frame"]) <= CONFIRM_WIN and o["slot"] == b["slot"] for o in out):
            out.append(b)
    out.sort(key=lambda h: h["frame"])
    merged = []  # 너무 가까운 두 타격은 하나로 (같은 선수)
    for h in out:
        if merged and merged[-1]["slot"] == h["slot"] and h["frame"] - merged[-1]["frame"] < min_gap_s * fps:
            continue
        merged.append(h)
    return merged


def _self_check():
    # 직선으로 날아가다 프레임 20에서 반대로 튕기는 공 + 오검출 + 몇 프레임 누락
    cands = []
    for i in range(40):
        x = 100 + 30 * i if i <= 20 else 100 + 30 * 20 - 30 * (i - 20)
        cands.append([(x, 400, 0.9)])
    cands[7] = [(900, 50, 0.8), (100 + 30 * 7, 400, 0.6)]   # 먼 곳의 가짜 후보가 conf가 더 높다
    cands[12] = [(460, 600, 0.7)]                            # 이동 거리는 가능하지만 직선에서 벗어난 가짜 점
    cands[13] = cands[14] = []                               # 누락
    tr = pick_track(cands)
    assert tr[7][:2] == (310, 400), tr[7]                    # 연속성으로 진짜를 고름
    assert tr[12] is not None                                # 연속성 검사는 통과해서 남아 있음
    tr = remove_spikes(tr)
    assert tr[12] is None                                    # 튀는 점 제거
    pts = interpolate(tr)
    assert all(p is not None for p in pts[10:16]) and pts[13]["interp"] and abs(pts[13]["x"] - 490) < 1
    turns = find_turns(pts)
    assert len(turns) == 1 and abs(turns[0]["frame"] - 20) <= 1, turns

    # 타격자 연결: 손목이 가까운 선수
    wr = lambda f: {0: [(700, 400)], 1: [(100, 100)]}
    hits = assign_hits(turns, wr)
    assert len(hits) == 1 and hits[0]["slot"] == 0

    # 스윙 합치기: 공이 쭉 보이는데 안 꺾이는 연습 스윙(프레임 3)은 버리고, 공이 안 보이던 구간의 스윙은 둔다
    pts2 = list(pts[:20]) + [None] * 20
    swings = [dict(frame=3, slot=1), dict(frame=33, slot=1), dict(frame=20, slot=0)]
    merged = merge_hits(swings, hits, pts2, 30)
    assert [(h["frame"], h["slot"]) for h in merged] == [(20, 0), (33, 1)], merged
    print("self-check OK")


if __name__ == "__main__":
    _self_check()
