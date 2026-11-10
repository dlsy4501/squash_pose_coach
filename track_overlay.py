"""
영상에서 선수를 추적해 중계 화면 스타일(overlay_ui.py) 오버레이를 입힌 mp4를 만든다.
선수 라벨/이름, 발밑 링, 타격 횟수와 타격 배너, 이동 거리, 코트맵.

코트 보정(바닥 라인 픽셀 <-> 코트 좌표 대응점)을 주면 호모그래피로 이동 거리/코트맵을 계산한다.
카메라가 고정(삼각대)일 때만 맞다. 보정이 없으면 이동 거리/코트맵 없이 나머지만 그린다.
일반 사용자는 app.py(웹 UI)를 쓰면 되고, 이 파일은 분석 엔진 + 개발자용 CLI다:
    python track_overlay.py --video v.mov --calib calib_img0433.json --start 144 --seconds 15
공 추적, 샷 종류, 득점 판정은 여기 없음.
"""
import argparse
import json
import math
import os
import subprocess

from ball_track import assign_hits, find_turns, interpolate, merge_hits, pick_track, remove_spikes
from overlay_ui import PLAYER_COLORS, Overlay
from rally import apply_winners, score_at, split_rallies

COURT_W, COURT_L = 6.4, 9.75  # m
SHORT_LINE = 5.44
MAX_STEP_M = 2.0   # 한 프레임에 이보다 많이 움직이면 ID 스위치/오검출로 보고 무시
EMA = 0.3
SWING_SPEED = 10.0  # 손목 속도(몸통길이/초, 3프레임 평균) 이 이상의 피크를 타격으로 봄
MIN_SHOT_GAP_S = 0.7
PEAK_LAG = 2       # 피크 확인에 쓰는 이후 프레임 수 (실시간 판정 지연)
MAX_PLAYERS = 2
PLAYER_HEIGHT_M = 1.7  # 보정 없을 때 이동 거리 환산용 (박스 높이 = 키)
APP_UPDATE_MAX, APP_EMA = 0.4, 0.05  # 옷 색 기억은 비슷한 때만, 천천히 갱신 (잘못 섞이면 영영 바뀌므로)
W_APP, W_POS, ID_BONUS = 0.4, 0.6, 0.1  # 슬롯 배정 비용: 옷 색 / 위치 / 트래커 ID가 같으면 깎아주는 값
BANNER_S = 0.9     # 타격 배너가 떠 있는 시간

# 보정 때 사용자가 클릭할 점들 (설명, 코트 좌표 m). 앞벽 왼쪽 바닥 모서리가 (0,0), x는 오른쪽, y는 뒤쪽.
CALIB_POINTS = [
    ("앞벽 바닥 왼쪽 모서리", (0.0, 0.0)),
    ("앞벽 바닥 오른쪽 모서리", (COURT_W, 0.0)),
    ("숏라인이 왼쪽 벽과 만나는 점", (0.0, SHORT_LINE)),
    ("숏라인이 오른쪽 벽과 만나는 점", (COURT_W, SHORT_LINE)),
    ("숏라인과 가운데 선이 만나는 점 (T자)", (COURT_W / 2, SHORT_LINE)),
]


class Mover:
    """한 선수의 코트 위 위치를 부드럽게 따라가며 이동 거리를 누적한다."""

    def __init__(self, max_step=MAX_STEP_M):
        self.max_step = max_step  # 보정 없이 픽셀 좌표를 쓸 때는 크게 준다
        self.pos = None
        self.dist = 0.0
        self.speed = 0.0
        self.trail = []
        self.raw = []
        self.speeds = []       # 프레임별 손목 속도(3프레임 평균: 키포인트 떨림 완화)
        self.wrists = None
        self.shots = []        # 타격 프레임 번호
        self.last_shot = -10**9

    def swing_speed(self, kpts_xy, kpts_conf, fps):
        """가장 빠른 손목의 속도를 몸통 길이 단위로 정규화(카메라 거리 무관). 못 구하면 0."""
        pts = {i: kpts_xy[i] for i in (9, 10) if kpts_conf[i] > 0.3}
        sh = [kpts_xy[i] for i in (5, 6)]
        hp = [kpts_xy[i] for i in (11, 12)]
        torso = math.dist(((sh[0][0] + sh[1][0]) / 2, (sh[0][1] + sh[1][1]) / 2),
                          ((hp[0][0] + hp[1][0]) / 2, (hp[0][1] + hp[1][1]) / 2))
        v = 0.0
        if self.wrists and torso > 1:
            v = max([math.dist(pts[i], self.wrists[i]) for i in pts if i in self.wrists] or [0]) / torso * fps
        self.wrists = pts
        return v

    def detect_shot(self, frame_idx, v, fps):
        """속도 피크(최근 PEAK_LAG 프레임 전이 앞뒤 구간 최대)를 타격으로 기록. 새 타격이면 True."""
        self.raw.append(v)
        self.speeds.append(sum(self.raw[-3:]) / len(self.raw[-3:]))
        w = self.speeds[-(2 * PEAK_LAG + 1):]
        if len(w) < 2 * PEAK_LAG + 1 or w[PEAK_LAG] < max(w) or w[PEAK_LAG] < SWING_SPEED:
            return False
        t = frame_idx - PEAK_LAG
        if t - self.last_shot < MIN_SHOT_GAP_S * fps:
            return False
        self.last_shot = t
        self.shots.append(t)
        return True

    def update(self, p, fps=30, scale=1.0):
        """scale: 좌표 단위 -> m 변환(보정 없을 때 화면 픽셀을 선수 키로 환산). 보정 있으면 1."""
        if self.pos is None:
            self.pos = p
        else:
            q = (self.pos[0] + EMA * (p[0] - self.pos[0]), self.pos[1] + EMA * (p[1] - self.pos[1]))
            step = math.dist(q, self.pos) * scale
            if step < self.max_step:
                self.dist += step
                self.speed += EMA * (step * fps - self.speed)
                self.pos = q
        self.trail.append(self.pos)
        return self.pos


def homography_from_pairs(px_pts, court_pts):
    """픽셀 점들과 대응하는 코트 좌표(m)로 호모그래피를 구한다 (4점 이상)."""
    import cv2
    import numpy as np
    if len(px_pts) < 4:
        raise ValueError("보정에는 점이 4개 이상 필요합니다")
    H, _ = cv2.findHomography(np.array(px_pts, dtype=np.float32), np.array(court_pts, dtype=np.float32))
    return H


def load_homography(path):
    pts = json.load(open(path, encoding="utf-8"))["points"]
    return homography_from_pairs([p["px"] for p in pts], [p["court"] for p in pts])


def to_court(H, xy):
    x, y = xy
    v = H @ [x, y, 1.0]
    return (float(v[0] / v[2]), float(v[1] / v[2]))


def foot_point(kpts_xy, kpts_conf, box):
    """양 발목 중간(신뢰도 낮으면 박스 아래 중앙)."""
    ok = [kpts_xy[i] for i in (15, 16) if kpts_conf[i] > 0.3]
    if ok:
        return (sum(p[0] for p in ok) / len(ok), sum(p[1] for p in ok) / len(ok))
    return ((box[0] + box[2]) / 2, box[3])


def in_court(c, margin=1.0):
    return -margin <= c[0] <= COURT_W + margin and -margin <= c[1] <= COURT_L + margin


def _iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(0.0, w) * max(0.0, h)
    return inter / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1e-6)


def _grow(b, r=0.2):
    """박스를 좌우로 r배 넓힌다 (바로 옆에 붙은 선수의 옷이 섞이는지 볼 때)."""
    w = (b[2] - b[0]) * r
    return (b[0] - w, b[1], b[2] + w, b[3])


def appearance(frame, xy, cf, box):
    """선수의 옷 색 특징: 상의(어깨~골반)와 하의(골반~발) 영역의 HSV 히스토그램을 이어 붙인 벡터."""
    import cv2
    import numpy as np
    x1, y1, x2, y2 = (int(v) for v in box)
    ok = lambda ids: all(cf[i] > 0.3 for i in ids)
    if ok((5, 6, 11, 12)):
        sy, hy = (xy[5][1] + xy[6][1]) / 2, (xy[11][1] + xy[12][1]) / 2
        xs = [xy[i][0] for i in (5, 6, 11, 12)]
        ux1, ux2 = min(xs), max(xs)
    else:  # 키포인트가 불안하면 박스 비율로 대체
        h = y2 - y1
        sy, hy, ux1, ux2 = y1 + 0.2 * h, y1 + 0.5 * h, x1 + 0.2 * (x2 - x1), x2 - 0.2 * (x2 - x1)
    ey = y2 - 0.05 * (y2 - y1)
    parts = []
    for ya, yb in ((sy, hy), (hy, ey)):
        a, b, c, d = int(max(0, ya)), int(max(0, yb)), int(max(0, ux1)), int(max(0, ux2))
        crop = frame[a:b, c:d]
        if crop.size < 30:
            parts.append(np.zeros(108, np.float32))
            continue
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1, 2], None, [12, 3, 3], [0, 180, 0, 256, 0, 256]).flatten()
        parts.append((h / max(h.sum(), 1e-6)).astype(np.float32))
    return parts


def appearance_dist(a, b):
    """두 옷 색 특징의 거리(0~1). 상의/하의 Bhattacharyya 거리의 평균."""
    import cv2
    return sum(cv2.compareHist(x, y, cv2.HISTCMP_BHATTACHARYYA) for x, y in zip(a, b)) / 2


def assign_slots(dets, slots, id_slot, box_h):
    """이번 프레임 검출(dets)을 선수 슬롯에 배정한다. dets: [{pid, foot, app}], slots: {슬롯: {foot, app}}.
    트래커 ID만 믿으면 두 선수가 겹쳤다 떨어질 때 바뀌므로, 옷 색+위치+(같은 ID면 보너스)의 비용이 최소인 조합을 고른다.
    반환: {검출 인덱스: 슬롯}. 슬롯보다 검출이 많으면 비용이 가장 낮은 쪽만 배정된다."""
    from itertools import permutations
    keys = list(slots)
    if len(dets) <= len(keys):  # 검출이 적으면 어느 슬롯에 넣을지가 선택지
        options = [list(zip(range(len(dets)), ks)) for ks in permutations(keys, len(dets))]
    else:
        options = [list(zip(ds, keys)) for ds in permutations(range(len(dets)), len(keys))]
    best, best_cost = {}, 1e9
    for pairs in options:
        cost = 0.0
        for d, k in pairs:
            sl, dt = slots[k], dets[d]
            app = appearance_dist(sl["app"], dt["app"]) if sl["app"] is not None else 0.5
            pos = min(1.0, math.dist(sl["foot"], dt["foot"]) / max(box_h, 1.0))
            cost += W_APP * app + W_POS * pos - (ID_BONUS if id_slot.get(dt["pid"]) == k else 0)
        if cost < best_cost:
            best, best_cost = dict(pairs), cost
    return best


def _encode(raw, src, start, seconds, out):
    """OpenCV가 쓴 mp4v(브라우저/휴대폰에서 재생 안 될 수 있음)를 H.264로 바꾸고 원본 소리를 붙인다."""
    import imageio_ffmpeg
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", raw,
           "-ss", str(start), "-t", str(seconds), "-i", src,
           "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-shortest", out]
    if subprocess.run(cmd, capture_output=True).returncode == 0:
        os.remove(raw)
    else:  # 변환 실패 시 소리 없는 원본 결과라도 남긴다
        os.replace(raw, out)


def detect_ball(model, frame, imgsz, conf=0.15):
    """공 후보 [(x, y, conf), ...] (박스 중심)."""
    r = model(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []
    return [(float((b[0] + b[2]) / 2), float((b[1] + b[3]) / 2), float(c))
            for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]


def analyze(video, H=None, start=0.0, seconds=None, ball_model=None, model_path="yolo11n-pose.pt"):
    """1단계: 영상을 한 번 훑어 프레임별 선수/공 후보를 모은다 (그리기는 안 함, 느린 단계).
    제너레이터: {"progress", "frame"(미리보기 BGR)}를 내다가 마지막에 {"analysis": data}.
    data["frames"][i] = {"p": [슬롯별 선수 {v, box, foot, pos, dist, speed, w}], "b": [공 후보]}"""
    import cv2
    from ultralytics import YOLO

    model = YOLO(model_path)
    ball = YOLO(ball_model) if ball_model else None
    ball_imgsz = (getattr(ball, "ckpt", None) or {}).get("train_args", {}).get("imgsz", 960) if ball else None
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    n_frames = min(int(seconds * fps) if seconds else total, total - int(start * fps))
    size = (int(cap.get(3)), int(cap.get(4)))

    movers, order, id_slot = {}, [], {}  # id_slot: 트래커 ID -> 마지막으로 배정된 슬롯(힌트일 뿐). order: 슬롯(처음 등장한 순서 = P1, P2)
    slot_state = {}  # 슬롯 -> {foot, app}: 마지막 위치와 옷 색(겹치지 않았을 때만 갱신)
    frames, swing_hits = [], []
    for fi in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        r = model.track(frame, persist=True, verbose=False)[0]
        cands = []
        if r.boxes is not None and r.boxes.id is not None:
            for box, pid, xy, cf in zip(r.boxes.xyxy.tolist(), r.boxes.id.int().tolist(),
                                        r.keypoints.xy.tolist(), r.keypoints.conf.tolist()):
                foot = foot_point(xy, cf, box)
                court = to_court(H, foot) if H is not None else foot
                if H is not None and not in_court(court):  # 관중/코트 밖 사람 제외
                    continue
                area = (box[2] - box[0]) * (box[3] - box[1])
                cands.append((area, box, pid, xy, cf, foot, court))
        cands.sort(key=lambda c: -c[0])  # 큰(가까운) 사람 우선
        cands = cands[:MAX_PLAYERS]

        dets = [dict(pid=c[2], foot=c[5], app=appearance(frame, c[3], c[4], c[1])) for c in cands]
        # 두 박스가 겹치면 옷 색이 섞이므로 그 프레임의 색은 기억에 반영하지 않는다
        apart = len(cands) < 2 or _iou(_grow(cands[0][1]), _grow(cands[1][1])) == 0
        chosen = assign_slots(dets, slot_state, id_slot, max([c[1][3] - c[1][1] for c in cands] or [1])) if slot_state else {}
        for di in range(len(cands)):  # 아직 슬롯이 모자라면 새 선수
            if di not in chosen and len(order) < MAX_PLAYERS:
                k = len(order)
                order.append(k)
                movers[k] = Mover(MAX_STEP_M if H is not None else 1e9)
                slot_state[k] = dict(foot=dets[di]["foot"], app=None)
                chosen[di] = k

        seen = {}
        for di, slot in chosen.items():
            _, box, pid, xy, cf, foot, court = cands[di]
            id_slot[pid] = slot
            st_ = slot_state[slot]
            if st_["app"] is None:
                st_["app"] = dets[di]["app"] if apart else None
            elif apart and appearance_dist(st_["app"], dets[di]["app"]) < APP_UPDATE_MAX:  # 확실히 같은 사람일 때만 갱신
                st_["app"] = [(1 - APP_EMA) * a + APP_EMA * b for a, b in zip(st_["app"], dets[di]["app"])]
            st_["foot"] = foot
            m = movers[slot]
            scale = 1.0 if H is not None else PLAYER_HEIGHT_M / max(box[3] - box[1], 1.0)
            m.update(court, fps, scale)
            wr = [list(xy[i]) for i in (9, 10) if cf[i] > 0.3]
            seen[slot] = dict(v=1, box=box, foot=list(foot), pos=list(m.pos), dist=m.dist, speed=m.speed, w=wr)
            if m.detect_shot(fi, m.swing_speed(xy, cf, fps), fps):
                swing_hits.append(dict(frame=fi - PEAK_LAG, slot=slot))
        players = []
        for slot in order:
            m = movers[slot]
            players.append(seen.get(slot) or dict(v=0, pos=list(m.pos), dist=m.dist, speed=0.0, w=[]))
        frames.append(dict(p=players, b=detect_ball(ball, frame, ball_imgsz) if ball else []))
        if fi % 15 == 0:
            prev = frame.copy()
            for sd in seen.values():
                cv2.rectangle(prev, tuple(int(v) for v in sd["box"][:2]), tuple(int(v) for v in sd["box"][2:]),
                              (80, 255, 80), 3)
            yield {"progress": fi / n_frames, "frame": cv2.resize(prev, None, fx=0.5, fy=0.5)}
    cap.release()
    yield {"analysis": dict(video=video, start=start, fps=fps, size=list(size), calibrated=H is not None,
                            has_ball=bool(ball), frames=frames, swing_hits=swing_hits)}


def postprocess(data, overrides=None):
    """2단계(빠름): 공 궤적 정리 -> 타격 확정 -> 랠리 -> 득점자. data에 points/hits/rallies를 채운다."""
    fps, scale = data["fps"], data["size"][1] / 1080
    frames = data["frames"]
    points = [None] * len(frames)
    ball_hits = []
    if data["has_ball"]:
        track = pick_track([f["b"] for f in frames], scale)
        points = interpolate(remove_spikes(track, scale))
        wrists = lambda i: {k: [tuple(w) for w in p["w"]] for k, p in enumerate(frames[i]["p"]) if p["v"]}
        ball_hits = assign_hits(find_turns(points, scale), wrists, scale)
    data["points"] = points
    data["hits"] = merge_hits(data["swing_hits"], ball_hits, points, fps) if ball_hits or data["has_ball"] \
        else [dict(h, source="swing") for h in data["swing_hits"]]
    data["rallies"] = apply_winners(split_rallies(data["hits"], fps), overrides)
    return data


def render(data, out, names=None):
    """3단계: 분석 결과(data)로 오버레이 영상을 만든다. 제너레이터: {"progress", "frame"} ... {"done", "out", "summary"}."""
    import cv2
    fps, size, start = data["fps"], tuple(data["size"]), data["start"]
    frames, hits, rallies, points = data["frames"], data["hits"], data["rallies"], data["points"]
    n_slots = max((len(f["p"]) for f in frames), default=0)  # 초반엔 선수가 1명만 보일 수 있다
    names = list(names or [])
    name_of = lambda k: (names[k] if k < len(names) and names[k].strip() else f"P{k + 1}")
    overlay = Overlay(size[1], [name_of(0), name_of(1)], data["calibrated"])
    cap = cv2.VideoCapture(data["video"])
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    raw = out + ".raw.mp4"
    writer = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    trails = [[] for _ in range(n_slots)]
    for fi, fr in enumerate(frames):
        ok, frame = cap.read()
        if not ok:
            break
        done = [h for h in hits if h["frame"] <= fi]
        banner = None
        if done and fi - done[-1]["frame"] <= BANNER_S * fps:
            h, k = done[-1], done[-1]["slot"]
            fade = (BANNER_S * fps - (fi - h["frame"])) / (0.3 * BANNER_S * fps)
            banner = dict(title=f"SHOT {len(done)}", sub=f"P{k + 1} · {name_of(k)}", alpha=min(1.0, fade),
                          color=PLAYER_COLORS[k % len(PLAYER_COLORS)])
        players = []
        for k, p in enumerate(fr["p"]):
            trails[k].append(tuple(p["pos"]))
            players.append(dict(name=name_of(k), color=PLAYER_COLORS[k % len(PLAYER_COLORS)],
                                box=p.get("box"), foot=p.get("foot"), dist=p["dist"], shots=sum(h["slot"] == k for h in done),
                                speed=p["speed"], trail=trails[k], visible=bool(p["v"])))
        tail = [(q["x"], q["y"]) for q in points[max(0, fi - 9):fi + 1] if q]
        ball = dict(trail=tail, now=points[fi] is not None) if tail else None
        score = score_at(rallies, fi, fps) if rallies else None
        writer.write(overlay.render(frame, dict(t=start + fi / fps, players=players, banner=banner, ball=ball, score=score)))
        if fi % 15 == 0:
            yield {"progress": fi / len(frames), "frame": cv2.resize(overlay.render(frame, dict(
                t=start + fi / fps, players=players, banner=banner, ball=ball, score=score)), None, fx=0.5, fy=0.5)}
    writer.release()
    cap.release()
    _encode(raw, data["video"], start, len(frames) / fps, out)
    last = lambda k: next(f["p"][k] for f in reversed(frames) if len(f["p"]) > k)
    summary = [dict(name=name_of(k), dist=last(k)["dist"], dist_estimated=not data["calibrated"],
                    shots=sum(h["slot"] == k for h in hits),
                    points=score_at(rallies, 10**9, fps)[k] if rallies else 0) for k in range(n_slots)]
    yield {"done": True, "out": out, "summary": summary}


def process_video(video, out, H=None, start=0.0, seconds=None, names=None, ball_model=None,
                  model_path="yolo11n-pose.pt", overrides=None):
    """분석 -> 후처리 -> 그리기를 이어 붙인 편의 함수 (제너레이터).
    진행 중엔 {"progress", "frame", "stage"}, 끝에 {"done", "out", "summary", "analysis"}."""
    data = None
    for ev in analyze(video, H, start, seconds, ball_model, model_path):
        if "analysis" in ev:
            data = ev["analysis"]
        else:
            yield dict(ev, progress=ev["progress"] * 0.7, stage="분석")
    postprocess(data, overrides)
    for ev in render(data, out, names):
        if ev.get("done"):
            yield dict(ev, analysis=data)
        else:
            yield dict(ev, progress=0.7 + ev["progress"] * 0.3, stage="영상 만드는 중")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--calib", help="보정 json (없으면 이동 거리/코트맵 없이 출력)")
    ap.add_argument("--out", default="overlay.mp4")
    ap.add_argument("--start", type=float, default=0, help="시작 초")
    ap.add_argument("--seconds", type=float, default=15)
    ap.add_argument("--names", nargs=2, help="선수 이름 2개")
    ap.add_argument("--ball-model", help="공 탐지 모델(.pt). 있으면 공 궤적으로 타격/랠리를 보정")
    a = ap.parse_args()
    H = load_homography(a.calib) if a.calib else None
    for ev in process_video(a.video, a.out, H, a.start, a.seconds, a.names, a.ball_model):
        if ev.get("done"):
            for s in ev["summary"]:
                dist = f'{"≈" if s["dist_estimated"] else ""}{s["dist"]:.1f} m, '
                print(f'{s["name"]}: {dist}타격 {s["shots"]}회, 득점 {s["points"]}')
            data = ev["analysis"]
            import report
            print("saved", ev["out"], "/", report.save_all(data, a.names or ["P1", "P2"], os.path.splitext(a.out)[0]))


def _self_check():
    # 호모그래피: 대응점이 정확히 코트 좌표로 돌아와야 한다
    pts = [((0, 0), (0, 0)), ((640, 0), (6.4, 0)), ((0, 500), (0, 5.0)), ((640, 500), (6.4, 5.0))]
    H = homography_from_pairs([p for p, _ in pts], [c for _, c in pts])
    for px, court in pts:
        assert math.dist(to_court(H, px), court) < 1e-3
    try:
        homography_from_pairs([(0, 0)] * 3, [(0, 0)] * 3)
        assert False
    except ValueError:
        pass

    # Mover: 정상 이동은 누적, 순간이동(ID 스위치)은 무시
    m = Mover()
    m.update((0, 0))
    for _ in range(50):
        m.update((1.0, 0))
    assert 0.9 < m.dist < 1.01
    d = m.dist
    m.update((6.0, 6.0))
    assert m.dist == d

    # in_court: 관중 위치는 걸러진다
    assert in_court((3, 5)) and not in_court((3, 15))

    # 타격 검출: 한 번 휘두르면 1회, 최소 간격 안의 두 번째 피크는 무시
    m = Mover()
    seq = [0, 0, 10, 30, 40, 30, 10, 0, 0, 0, 0, 40, 60, 40, 0, 0, 0]
    hits = [m.detect_shot(i, v, 30) for i, v in enumerate(seq)]
    assert sum(hits) == 1 and 3 <= m.shots[0] <= 6, m.shots
    # 슬롯 배정: 옷 색이 다르면 트래커 ID가 서로 바뀌어도 원래 선수에게 돌려준다
    import numpy as np
    img = np.zeros((200, 100, 3), np.uint8)
    xy = [(30, 50)] * 17
    xy[5], xy[6], xy[11], xy[12] = (20, 50), (80, 50), (25, 100), (75, 100)
    cf = [1.0] * 17

    def person(shirt, shorts):
        f = img.copy()
        f[50:100] = shirt
        f[100:190] = shorts
        return appearance(f, xy, cf, (0, 20, 100, 200))

    red, blue = person((0, 0, 255), (40, 40, 40)), person((255, 0, 0), (200, 200, 200))
    slots = {0: dict(foot=(100, 500), app=red), 1: dict(foot=(110, 500), app=blue)}
    dets = [dict(pid=7, foot=(105, 500), app=blue), dict(pid=8, foot=(104, 500), app=red)]  # 위치는 거의 같음
    assert assign_slots(dets, slots, {7: 0, 8: 1}, 200) == {0: 1, 1: 0}  # ID는 바뀌었어도 색으로 복구
    assert assign_slots(dets[:1], slots, {7: 0}, 200) == {0: 1}  # 한 명만 보여도 옷 색이 맞는 슬롯으로
    print("self-check OK")


if __name__ == "__main__":
    import sys
    _self_check() if len(sys.argv) == 1 else main()
