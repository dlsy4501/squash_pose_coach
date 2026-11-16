"""
스윙 1회 구간을 자동으로 잡고(백스윙 탑 ~ 임팩트), 그 구간의 자세 지표를 뽑는다.
track_overlay.py의 analyze()가 frames[i]["p"][slot]에 저장해 둔 kp(관절 좌표)와
sw(손목 스윙 속도, 프레임별)를 그대로 쓴다. 포즈 모델을 다시 돌리지 않는다.

지금 나오는 지표: 백스윙 높이, 어깨 회전(근사), 어깨 기울기, 무릎 각도, 몸 기울기,
스탠스 너비, 스윙 시간. 라켓 속도(mph)와 매끄러움(smoothness)은 여기 없음
(라켓 자체를 추적하지 않고, 구간 전체 궤적의 가속도 분석이 더 필요해서 다음 단계).

한 줄 코칭 피드백은 지금은 경험적으로 정한 "정상 범위"를 벗어난 지표 중 가장 많이
벗어난 것 하나를 고른다 - 레퍼런스 샘플 기반 기준(build_reference.py 식)으로 바꾸는 게
다음 단계(점수화)의 목표.

사용법 (코드에서):
    swings = find_swings(frames, slot, hits, fps)
    for s in swings:
        print(s["metrics"], s["tip"])
"""
import math

# pose_compare.py와 같은 COCO keypoint 인덱스
L_SH, R_SH = 5, 6
L_EL, R_EL = 7, 8
L_WR, R_WR = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANK, R_ANK = 15, 16

TOP_LOOKBACK_S = 1.2    # 백스윙 탑을 찾을 때 임팩트에서 거슬러 올라가는 최대 시간
MIN_CONF = 0.3

# ponytail: 전부 경험적으로 정한 "정상 범위". 실제 레퍼런스 샘플이 쌓이면
# build_reference.py처럼 평균±표준편차 기준으로 바꿔야 함 (지금은 placeholder)
OK_RANGE = {
    "무릎각도": (110, 150),        # 너무 펴지면(>150) "더 낮게", 너무 굽으면(<110) 과하게 낮다고 봄
    "몸기울기": (0, 25),           # 수직 기준 기울어진 각도, 클수록 과하게 기울어짐
    "스탠스너비": (0.9, 1.8),      # 발목 간격 / 어깨너비
    "어깨회전": (0.08, 10),        # 방향비율 변화량(backswing 중 몸통 대비 어깨너비 변화) - 작으면 회전 부족
    "백스윙높이": (0.15, 10),      # 탑에서 손목이 어깨보다 위로 올라간 정도(몸통 길이 비율)
}
TIPS = {
    "무릎각도_low": "무릎이 너무 많이 굽었어요. 조금만 펴서 균형을 잡아보세요",
    "무릎각도_high": "무릎을 더 굽혀서 낮고 안정적인 자세를 만들어보세요",
    "몸기울기_high": "상체가 많이 기울었어요. 중심을 세워서 쳐보세요",
    "스탠스너비_low": "보폭이 좁아요. 발을 조금 더 벌려 안정적인 자세를 만들어보세요",
    "스탠스너비_high": "보폭이 너무 넓어요. 조금 좁혀서 회복이 빠르게 되도록 해보세요",
    "어깨회전_low": "어깨를 더 돌려서 백스윙을 크게 가져가보세요",
    "백스윙높이_low": "라켓을 더 높이 들어올리는 백스윙을 해보세요",
}


def _pt(kp, idx):
    v = kp.get(str(idx))
    if not v or v[2] < MIN_CONF:
        return None
    return (v[0], v[1])


def angle(a, b, c):
    """b를 꼭짓점으로 하는 a-b-c 각도(도)."""
    if a is None or b is None or c is None:
        return None
    v1, v2 = (a[0] - b[0], a[1] - b[1]), (c[0] - b[0], c[1] - b[1])
    mag = math.hypot(*v1) * math.hypot(*v2)
    if mag == 0:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / mag))))


def orientation_ratio(kp):
    """pose_compare.compute_orientation과 동일: 어깨너비/몸통길이. 방향(카메라를 얼마나
    정면으로 보는지) 근사치 - 몸통을 틀수록(회전할수록) 어깨가 좁아 보여서 값이 작아진다."""
    l_sh, r_sh, l_hip, r_hip = _pt(kp, L_SH), _pt(kp, R_SH), _pt(kp, L_HIP), _pt(kp, R_HIP)
    if not (l_sh and r_sh and l_hip and r_hip):
        return None
    sw = math.hypot(r_sh[0] - l_sh[0], r_sh[1] - l_sh[1])
    mid_sh = ((l_sh[0] + r_sh[0]) / 2, (l_sh[1] + r_sh[1]) / 2)
    mid_hip = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)
    torso = math.hypot(mid_sh[0] - mid_hip[0], mid_sh[1] - mid_hip[1])
    return sw / torso if torso else None


def torso_length(kp):
    mid_sh = _mid(kp, L_SH, R_SH)
    mid_hip = _mid(kp, L_HIP, R_HIP)
    if not (mid_sh and mid_hip):
        return None
    return math.hypot(mid_sh[0] - mid_hip[0], mid_sh[1] - mid_hip[1])


def _mid(kp, i, j):
    a, b = _pt(kp, i), _pt(kp, j)
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2) if (a and b) else None


def find_top(frames, slot, hit_frame, fps, lookback_s=TOP_LOOKBACK_S):
    """hit_frame 이전 lookback_s 안에서 스윙 속도(sw)가 가장 낮은 지점 = 백스윙이 멈췄다가
    내려오기 시작하는 지점(백스윙 탑)으로 본다. 못 찾으면 hit_frame 그대로(스윙시간 0)."""
    lo = max(0, hit_frame - int(lookback_s * fps))
    best_f, best_v = hit_frame, None
    for f in range(lo, hit_frame):
        p = frames[f]["p"][slot] if slot < len(frames[f]["p"]) else None
        if not p or not p.get("v"):
            continue
        v = p.get("sw", 0.0)
        if best_v is None or v < best_v:
            best_f, best_v = f, v
    return best_f


def swing_metrics(frames, slot, top_frame, hit_frame, fps):
    """[top_frame, hit_frame] 구간의 자세 지표를 계산. 못 구한 값은 None."""
    def kp_at(f):
        p = frames[f]["p"][slot] if 0 <= f < len(frames) and slot < len(frames[f]["p"]) else None
        return p.get("kp") if p and p.get("v") else None

    top_kp, hit_kp = kp_at(top_frame), kp_at(hit_frame)
    m = {}
    m["스윙시간_s"] = round((hit_frame - top_frame) / fps, 2)

    if hit_kp:
        knee = [a for a in (angle(_pt(hit_kp, L_HIP), _pt(hit_kp, L_KNEE), _pt(hit_kp, L_ANK)),
                            angle(_pt(hit_kp, R_HIP), _pt(hit_kp, R_KNEE), _pt(hit_kp, R_ANK))) if a is not None]
        m["무릎각도_deg"] = round(min(knee), 1) if knee else None  # 더 굽은 쪽(각도 작은 쪽) 기준

        mid_sh, mid_hip = _mid(hit_kp, L_SH, R_SH), _mid(hit_kp, L_HIP, R_HIP)
        if mid_sh and mid_hip:
            dx, dy = mid_sh[0] - mid_hip[0], mid_hip[1] - mid_sh[1]  # 이미지 y는 아래로 증가하니 뒤집어서 "위" 기준
            m["몸기울기_deg"] = round(math.degrees(math.atan2(abs(dx), max(dy, 1e-6))), 1)
        else:
            m["몸기울기_deg"] = None

        l_ank, r_ank, l_sh, r_sh = _pt(hit_kp, L_ANK), _pt(hit_kp, R_ANK), _pt(hit_kp, L_SH), _pt(hit_kp, R_SH)
        if l_ank and r_ank and l_sh and r_sh:
            sw_w = math.hypot(r_sh[0] - l_sh[0], r_sh[1] - l_sh[1])
            m["스탠스너비_ratio"] = round(math.hypot(r_ank[0] - l_ank[0], r_ank[1] - l_ank[1]) / sw_w, 2) if sw_w else None
        else:
            m["스탠스너비_ratio"] = None
    else:
        m.update({"무릎각도_deg": None, "몸기울기_deg": None, "스탠스너비_ratio": None})

    o_top, o_hit = orientation_ratio(top_kp) if top_kp else None, orientation_ratio(hit_kp) if hit_kp else None
    m["어깨회전_변화량"] = round(abs(o_top - o_hit), 2) if (o_top is not None and o_hit is not None) else None

    if top_kp:
        torso = torso_length(top_kp)
        wrists = [p for p in (_pt(top_kp, L_WR), _pt(top_kp, R_WR)) if p]
        mid_sh = _mid(top_kp, L_SH, R_SH)
        if torso and wrists and mid_sh:
            highest = min(w[1] for w in wrists)  # y가 작을수록 화면 위쪽
            m["백스윙높이_ratio"] = round((mid_sh[1] - highest) / torso, 2)
        else:
            m["백스윙높이_ratio"] = None
    else:
        m["백스윙높이_ratio"] = None

    return m


def coaching_tip(metrics):
    """OK_RANGE를 가장 많이 벗어난 지표 하나를 골라 한 줄 피드백. 다 정상 범위면 칭찬 한 줄."""
    key_for = {"무릎각도_deg": "무릎각도", "몸기울기_deg": "몸기울기", "스탠스너비_ratio": "스탠스너비",
              "어깨회전_변화량": "어깨회전", "백스윙높이_ratio": "백스윙높이"}
    worst, worst_over, direction = None, 0.0, None
    for mkey, okkey in key_for.items():
        v = metrics.get(mkey)
        if v is None or okkey not in OK_RANGE:
            continue
        lo, hi = OK_RANGE[okkey]
        over = max(lo - v, v - hi, 0.0) / max(hi - lo, 1e-6)  # 범위 폭 대비 얼마나 벗어났는지
        if over > worst_over:
            worst, worst_over, direction = okkey, over, ("low" if v < lo else "high")
    if worst is None or worst_over < 0.15:  # 살짝 벗어난 건 무시(ponytail: 15% 임계값도 placeholder)
        return "이 스윙은 눈에 띄는 문제 없이 좋아요"
    return TIPS.get(f"{worst}_{direction}", f"{worst}이(가) 기준에서 많이 벗어났어요")


def find_swings(frames, slot, hits, fps):
    """hits(merge_hits 결과) 중 이 slot 타격들에 대해 구간+지표+코칭을 묶어 리스트로.
    결과 각 항목: {hit_frame, top_frame, metrics, tip}"""
    out = []
    for h in hits:
        if h["slot"] != slot:
            continue
        top = find_top(frames, slot, h["frame"], fps)
        metrics = swing_metrics(frames, slot, top, h["frame"], fps)
        out.append(dict(hit_frame=h["frame"], top_frame=top, metrics=metrics, tip=coaching_tip(metrics)))
    return out


def _self_check():
    # 가짜 선수 1명짜리 프레임을 만든다: 백스윙 탑(손목이 어깨 위로 높이, 속도 느림) ->
    # 임팩트(무릎 굽히고 체중 이동, 속도 빠름) 구간.
    def kp(l_wrist_y, knee_angle_l=130, lean_dx=0.0, stance=50.0):
        # 허벅지는 수직으로 고정, 정강이 각도만 삼각함수로 조절해 무릎각도를 정확히 맞춘다
        # (knee->hip은 (0,-1) 방향으로 고정, knee->ankle이 그 방향과 knee_angle_l도를 이루게)
        mid_x, sh_width, thigh, shin = 50.0, 40.0, 40.0, 40.0
        l_sh, r_sh = (mid_x - sh_width / 2, 100.0), (mid_x + sh_width / 2, 100.0)
        l_hip, r_hip = (mid_x - 15 + lean_dx, 160.0), (mid_x + 15 + lean_dx, 160.0)
        l_knee, r_knee = (l_hip[0], l_hip[1] + thigh), (r_hip[0], r_hip[1] + thigh)
        rad = math.radians(knee_angle_l)
        l_ank = (l_knee[0] + shin * math.sin(rad) - stance / 2 + 25,
                l_knee[1] - shin * math.cos(rad))
        r_ank = (r_knee[0] + shin * math.sin(rad) + stance / 2 - 25,
                r_knee[1] - shin * math.cos(rad))
        full = [(0, 0)] * 17
        for i, p in ((L_SH, l_sh), (R_SH, r_sh), (L_HIP, l_hip), (R_HIP, r_hip),
                    (L_KNEE, l_knee), (R_KNEE, r_knee), (L_ANK, l_ank), (R_ANK, r_ank),
                    (L_WR, (mid_x - 10, l_wrist_y)), (R_WR, (mid_x + 10, l_wrist_y)),
                    (L_EL, (mid_x, 70.0)), (R_EL, (mid_x, 70.0))):
            full[i] = p
        return {str(i): [p[0], p[1], 1.0] for i, p in enumerate(full)}

    fps = 30
    frames = []
    for f in range(40):
        if f < 15:
            sw, wrist_y = 1.0, 100 - f * 3
        elif f < 20:
            sw, wrist_y = 0.2, 55
        else:
            sw, wrist_y = min(30.0, (f - 19) * 6), 55 + (f - 19) * 3
        is_hit_region = f >= 23
        this_kp = kp(wrist_y, knee_angle_l=95 if is_hit_region else 150, lean_dx=18 if is_hit_region else 0)
        frames.append(dict(p=[dict(v=1, kp=this_kp, sw=sw)]))

    hits = [dict(frame=25, slot=0)]
    top = find_top(frames, 0, 25, fps)
    assert 15 <= top <= 21, top

    swings = find_swings(frames, 0, hits, fps)
    assert len(swings) == 1
    m = swings[0]["metrics"]
    assert m["스윙시간_s"] > 0
    assert m["무릎각도_deg"] is not None and m["무릎각도_deg"] < 110
    assert m["몸기울기_deg"] is not None and m["몸기울기_deg"] > 10
    assert m["백스윙높이_ratio"] is not None and m["백스윙높이_ratio"] > 0
    assert "무릎" in swings[0]["tip"] or "굽" in swings[0]["tip"] or "기울" in swings[0]["tip"], swings[0]["tip"]

    ok_metrics = {"무릎각도_deg": 130, "몸기울기_deg": 5, "스탠스너비_ratio": 1.2,
                 "어깨회전_변화량": 0.2, "백스윙높이_ratio": 0.5}
    assert "문제" in coaching_tip(ok_metrics)
    print("self-check OK")


if __name__ == "__main__":
    _self_check()
