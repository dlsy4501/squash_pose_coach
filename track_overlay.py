"""
영상에서 선수를 추적해 P1/P2 라벨, 발밑 원, 코트맵, 이동 거리를 입힌 mp4를 만든다.
바닥 라인의 픽셀<->코트 좌표 대응점(calib json)으로 호모그래피를 한 번 구해 쓰기 때문에
카메라가 고정(삼각대)일 때만 맞다. 핸드헬드로 흔들리면 코트맵/거리가 틀어진다.

사용법:
    python track_overlay.py --video v.mov --calib calib_img0433.json --start 144 --seconds 15
공 추적, 샷 판정, 득점 판정은 여기 없음.
"""
import argparse
import json
import math

COURT_W, COURT_L = 6.4, 9.75  # m
SHORT_LINE = 5.44
MAX_STEP_M = 2.0   # 한 프레임에 이보다 많이 움직이면 ID 스위치/오검출로 보고 무시
EMA = 0.3
SWING_SPEED = 10.0  # 손목 속도(몸통길이/초, 3프레임 평균) 이 이상의 피크를 타격으로 봄
MIN_SHOT_GAP_S = 0.7
MAX_PLAYERS = 2
PEAK_LAG = 2       # 피크 확인에 쓰는 이후 프레임 수 (실시간 판정 지연)
COLORS = [(80, 255, 80), (255, 200, 60), (60, 120, 255), (255, 80, 200)]  # BGR


class Mover:
    """한 선수의 코트 위 위치를 부드럽게 따라가며 이동 거리를 누적한다."""

    def __init__(self):
        self.pos = None
        self.dist = 0.0
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

    def update(self, p):
        if self.pos is None:
            self.pos = p
        else:
            q = (self.pos[0] + EMA * (p[0] - self.pos[0]), self.pos[1] + EMA * (p[1] - self.pos[1]))
            step = math.dist(q, self.pos)
            if step < MAX_STEP_M:
                self.dist += step
                self.pos = q
        self.trail.append(self.pos)
        return self.pos


def load_homography(path):
    import cv2
    import numpy as np
    pts = json.load(open(path, encoding="utf-8"))["points"]
    src = np.array([p["px"] for p in pts], dtype=np.float32)
    dst = np.array([p["court"] for p in pts], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    return H


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


def draw_map(frame, movers, order, w=280):
    import cv2
    h = int(w * COURT_L / COURT_W)
    H_, W_ = frame.shape[:2]
    x0, y0 = W_ - w - 20, H_ - h - 20
    cv2.rectangle(frame, (x0 - 6, y0 - 6), (x0 + w + 6, y0 + h + 6), (30, 30, 30), -1)
    s = w / COURT_W
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (255, 255, 255), 1)
    ys = y0 + int(SHORT_LINE * s)
    cv2.line(frame, (x0, ys), (x0 + w, ys), (255, 255, 255), 1)
    cv2.line(frame, (x0 + w // 2, ys), (x0 + w // 2, y0 + h), (255, 255, 255), 1)
    for k, pid in enumerate(order):
        m, col = movers[pid], COLORS[k % len(COLORS)]
        pts = [(x0 + int(p[0] * s), y0 + int(p[1] * s)) for p in m.trail[-90:]]
        for a, b in zip(pts, pts[1:]):
            cv2.line(frame, a, b, col, 1)
        if pts:
            cv2.circle(frame, pts[-1], 6, col, -1)


def draw_frame(frame, players, movers, order, flash=None):
    import cv2
    for pid, (box, foot) in players.items():
        k = order.index(pid)
        col = COLORS[k % len(COLORS)]
        x1, y1 = int(box[0]), int(box[1])
        cv2.putText(frame, f"P{k + 1}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 3)
        cv2.ellipse(frame, (int(foot[0]), int(foot[1])), (50, 16), 0, 0, 360, col, 3)
    for k, pid in enumerate(order):
        cv2.putText(frame, f"P{k + 1}  {movers[pid].dist:5.1f} m  shots {len(movers[pid].shots)}", (30, 60 + 50 * k),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLORS[k % len(COLORS)], 3)
    if flash:
        cv2.putText(frame, flash, (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 4)
    draw_map(frame, movers, order)


def main():
    import cv2
    from ultralytics import YOLO

    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", default="overlay.mp4")
    ap.add_argument("--start", type=float, default=0, help="시작 초")
    ap.add_argument("--seconds", type=float, default=15)
    ap.add_argument("--model", default="yolo11n-pose.pt")
    a = ap.parse_args()

    H = load_homography(a.calib)
    model = YOLO(a.model)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_MSEC, a.start * 1000)
    size = (int(cap.get(3)), int(cap.get(4)))
    out = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    movers, order, alias = {}, [], {}  # alias: 트래커 ID -> 선수 슬롯. order: 처음 등장한 순서대로 P1, P2...
    flash, flash_until, n_shots = None, -1, 0
    for fi in range(int(a.seconds * fps)):
        ok, frame = cap.read()
        if not ok:
            break
        r = model.track(frame, persist=True, verbose=False)[0]
        players = {}
        if r.boxes is not None and r.boxes.id is not None:
            for box, pid, xy, cf in zip(r.boxes.xyxy.tolist(), r.boxes.id.int().tolist(),
                                        r.keypoints.xy.tolist(), r.keypoints.conf.tolist()):
                foot = foot_point(xy, cf, box)
                court = to_court(H, foot)
                if not in_court(court):  # 관중/코트 밖 사람 제외
                    continue
                if pid not in alias:
                    if len(order) < MAX_PLAYERS:
                        alias[pid] = pid
                        movers[pid] = Mover()
                        order.append(pid)
                    else:  # 트래커가 ID를 새로 줬을 때: 지금 안 보이는 선수 중 마지막 위치가 가장 가까운 슬롯에 잇는다
                        free = [q for q in order if q not in players]
                        if not free:
                            continue
                        alias[pid] = min(free, key=lambda q: math.dist(movers[q].pos, court))
                slot = alias[pid]
                if slot in players:
                    continue
                m = movers[slot]
                m.update(court)
                players[slot] = (box, foot)
                if m.detect_shot(fi, m.swing_speed(xy, cf, fps), fps):
                    n_shots += 1
                    flash, flash_until = f"SHOT {n_shots}  P{order.index(slot) + 1}", fi + int(fps * 0.7)
        draw_frame(frame, players, movers, order, flash if fi <= flash_until else None)
        out.write(frame)
    out.release()
    for k, pid in enumerate(order):
        print(f"P{k + 1} (track {pid}): {movers[pid].dist:.1f} m, 타격 {len(movers[pid].shots)}회 "
              f"(영상 {[round(a.start + t / fps, 1) for t in movers[pid].shots]}초)")
    print("saved", a.out)


def _self_check():
    import numpy as np
    import cv2
    # 호모그래피: 대응점이 정확히 코트 좌표로 돌아와야 한다
    pts = [((0, 0), (0, 0)), ((640, 0), (6.4, 0)), ((0, 500), (0, 5.0)), ((640, 500), (6.4, 5.0))]
    H, _ = cv2.findHomography(np.array([p for p, _ in pts], np.float32), np.array([c for _, c in pts], np.float32))
    for px, court in pts:
        got = to_court(H, px)
        assert math.dist(got, court) < 1e-3, (got, court)

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
