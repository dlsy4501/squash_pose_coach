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
import rally
from rally import apply_winners, score_at, split_rallies

COURT_W, COURT_L = 6.4, 9.75  # m
SHORT_LINE = 5.44
MAX_STEP_M = 2.0   # 한 프레임에 이보다 많이 움직이면 ID 스위치/오검출로 보고 무시
EMA = 0.3
SWING_SPEED = 10.0  # 손목 속도(몸통길이/초, 3프레임 평균) 이 이상의 피크를 타격으로 봄
MIN_SHOT_GAP_S = 0.7
PEAK_LAG = 2       # 피크 확인에 쓰는 이후 프레임 수 (실시간 판정 지연)
MAX_PLAYERS = 2
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

    def update(self, p, fps=30):
        if self.pos is None:
            self.pos = p
        else:
            q = (self.pos[0] + EMA * (p[0] - self.pos[0]), self.pos[1] + EMA * (p[1] - self.pos[1]))
            step = math.dist(q, self.pos)
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

    movers, order, alias = {}, [], {}  # alias: 트래커 ID -> 선수 슬롯. order: 슬롯(처음 등장한 순서 = P1, P2)
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

        seen = {}
        for _, box, pid, xy, cf, foot, court in cands[:MAX_PLAYERS]:
            if pid not in alias:
                if len(order) < MAX_PLAYERS:
                    alias[pid] = pid
                    movers[pid] = Mover(MAX_STEP_M if H is not None else 1e9)
                    order.append(pid)
                else:  # 트래커가 ID를 새로 줬을 때: 지금 안 보이는 선수 중 마지막 위치가 가장 가까운 슬롯에 잇는다
                    free = [q for q in order if q not in seen]
                    if not free:
                        continue
                    alias[pid] = min(free, key=lambda q: math.dist(movers[q].pos, court))
            slot = alias[pid]
            if slot in seen:
                continue
            m = movers[slot]
            m.update(court, fps)
            wr = [list(xy[i]) for i in (9, 10) if cf[i] > 0.3]
            seen[slot] = dict(v=1, box=box, foot=list(foot), pos=list(m.pos), dist=m.dist, speed=m.speed, w=wr)
            if m.detect_shot(fi, m.swing_speed(xy, cf, fps), fps):
                swing_hits.append(dict(frame=fi - PEAK_LAG, slot=order.index(slot)))
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
    summary = [dict(name=name_of(k), dist=last(k)["dist"] if data["calibrated"] else None,
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
                dist = f'{s["dist"]:.1f} m, ' if s["dist"] is not None else ""
                print(f'{s["name"]}: {dist}타격 {s["shots"]}회, 득점 {s["points"]}')
            data = ev["analysis"]
            csv_path = os.path.splitext(a.out)[0] + "_rallies.csv"
            rally.save_csv(data["rallies"], data["fps"], csv_path, a.start, a.names or ["P1", "P2"])
            print("saved", ev["out"], "/", csv_path)


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
    print("self-check OK")


if __name__ == "__main__":
    import sys
    _self_check() if len(sys.argv) == 1 else main()
