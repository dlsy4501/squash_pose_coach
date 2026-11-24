"""
라켓 헤드 키포인트 라벨링 도구. 영상에서 프레임을 무작위로 뽑아, 라켓 헤드 테두리에 점 4개를
순서대로 찍으면(팁 -> 오른쪽 -> 목(throat) -> 왼쪽) YOLO-pose 형식 라벨로 저장한다.
한 프레임에 선수가 둘이면 라켓도 보통 둘이므로, 4점을 두 번(총 8번) 찍을 수 있다.
(racket_dataset/images/*.jpg, racket_dataset/labels/*.txt, 클래스 0 = racket,
한 줄에 라켓 하나: "0 cx cy bw bh  tip_x tip_y v  right_x right_y v  throat_x throat_y v  left_x left_y v")

박스 드래그 방식은 라켓이 있다/없다, 어디 있다는 알 수 있지만 "라켓 면이 얼마나 열렸는지"(스윙 중
타격면의 카메라 대비 기울기)는 알 수 없다. 헤드는 거의 타원이므로 테두리의 긴 축(팁-목)과 짧은 축
(왼쪽-오른쪽) 끝점 4개를 찍으면, 두 축의 길이 비율로 타원이 얼마나 눌려 보이는지(= 면이 카메라 쪽으로
얼마나 열려 있는지) 추정할 수 있다. 비율 계산은 racket_face.py.

사용법:
    python label_racket.py --video IMG_0433.MOV IMG_7867.MOV --n 150   # 영상 여러 개 가능
키: 클릭 4번=라켓 1개(순서: 팁 -> 오른쪽 -> 목 -> 왼쪽, 한 프레임에 여러 번 가능, 보통 선수 수만큼)
    / u=마지막 점(또는 라켓) 취소 / n 또는 엔터=이 프레임 완료하고 다음(점 4개 안 채운 라켓은 버려짐)
    / s=라켓이 안 보임(빈 라벨) / b=이전 프레임으로 / q=종료
이미 라벨된 프레임은 건너뛰므로 q로 나갔다가 같은 명령으로 이어서 할 수 있다. b로 되돌아간 프레임은
기존 점을 불러와서 보여주므로 추가/취소 후 n으로 다시 저장하면 된다.
영상에서 프레임을 뽑는 부분(extract_frames)과 라벨 파일 경로 규칙은 label_ball.py와 공유한다.
"""
import argparse
import os

import numpy as np

from label_ball import extract_frames, label_path

POINT_NAMES = ("팁", "오른쪽", "목", "왼쪽")  # 찍는 순서: 헤드 긴 축(팁-목) 다음 짧은 축(오른쪽-왼쪽)


def yolo_kpt_line(points, w, h):
    """라켓 1개의 4점(픽셀, 팁/오른쪽/목/왼쪽 순) -> YOLO-pose 한 줄(정규화)."""
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
    parts = [f"0 {cx / w:.6f} {cy / h:.6f} {bw / w:.6f} {bh / h:.6f}"]
    for x, y in points:
        parts.append(f"{x / w:.6f} {y / h:.6f} 2")
    return " ".join(parts) + "\n"


def yolo_kpt_lines(rackets, w, h):
    """[[(x,y), (x,y), (x,y), (x,y)], ...] (픽셀) -> YOLO-pose 여러 줄. 없으면 빈 문자열(라켓 없음)."""
    return "".join(yolo_kpt_line(points, w, h) for points in rackets)


def load_rackets(img_path, lbl_dir, w, h):
    """기존 라벨 파일 -> [[(x,y), (x,y), (x,y), (x,y)], ...] (픽셀). 없으면 빈 리스트."""
    p = label_path(img_path, lbl_dir)
    if not os.path.exists(p):
        return []
    rackets = []
    with open(p) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 17:  # class + bbox(4) + 4점*(x,y,v)
                continue
            vals = [float(v) for v in parts[5:17]]
            points = [(vals[i] * w, vals[i + 1] * h) for i in range(0, 12, 3)]
            rackets.append(points)
    return rackets


def run_ui(img_dir, lbl_dir, scale):
    import cv2
    os.makedirs(lbl_dir, exist_ok=True)
    imgs = sorted(os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".jpg"))
    todo = [i for i, p in enumerate(imgs) if not os.path.exists(label_path(p, lbl_dir))]
    if not todo:
        print("라벨할 프레임이 없습니다")
        return
    i = todo[0]
    clicks = []  # 이 프레임에서 찍은 점(픽셀), 4개씩 끊어서 라켓 하나

    def on_mouse(event, mx, my, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((mx / scale, my / scale))

    cv2.namedWindow("label_racket")
    cv2.setMouseCallback("label_racket", on_mouse)
    while 0 <= i < len(imgs):
        img = cv2.imread(imgs[i])
        h, w = img.shape[:2]
        loaded = load_rackets(imgs[i], lbl_dir, w, h)
        clicks.clear()
        for points in loaded:
            clicks.extend(points)
        done = sum(os.path.exists(label_path(p, lbl_dir)) for p in imgs)
        next_i = None
        while next_i is None:
            show = cv2.resize(img, None, fx=scale, fy=scale)
            n_full = len(clicks) // 4
            expect = POINT_NAMES[len(clicks) % 4]
            cv2.putText(show, f"{i + 1}/{len(imgs)}  labeled {done}  racket {n_full}  next={expect}"
                               "  click=point  u=undo  n/enter=next  s=none  b=back  q=quit",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            for g in range(0, len(clicks) - len(clicks) % 4, 4):
                poly = [(int(x * scale), int(y * scale)) for x, y in clicks[g:g + 4]]
                cv2.polylines(show, [np.array(poly, dtype=np.int32)], True, (0, 255, 0), 2)
                for x, y in poly:
                    cv2.circle(show, (x, y), 4, (0, 255, 0), -1)
            for x, y in clicks[len(clicks) - len(clicks) % 4:]:
                cv2.circle(show, (int(x * scale), int(y * scale)), 4, (0, 200, 255), -1)
            cv2.imshow("label_racket", show)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("u") and clicks:
                clicks.pop()
            elif key == ord("s"):
                clicks.clear()
                open(label_path(imgs[i], lbl_dir), "w").close()
                next_i = i + 1
            elif key in (ord("n"), 13):
                full = len(clicks) - len(clicks) % 4
                rackets = [clicks[g:g + 4] for g in range(0, full, 4)]
                with open(label_path(imgs[i], lbl_dir), "w") as f:
                    f.write(yolo_kpt_lines(rackets, w, h))
                next_i = i + 1
            elif key == ord("b"):
                next_i = max(0, i - 1)
            elif key == ord("q"):
                cv2.destroyAllWindows()
                return
        i = next_i
    cv2.destroyAllWindows()
    print("모든 프레임 라벨 완료")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", nargs="+", help="영상 파일 (여러 개 가능)")
    ap.add_argument("--n", type=int, default=300, help="영상 하나당 뽑을 프레임 수")
    ap.add_argument("--out", default="racket_dataset")
    ap.add_argument("--scale", type=float, default=0.75, help="화면 표시 배율")
    a = ap.parse_args()
    if not a.video:
        ap.error("--video 가 필요합니다")
    img_dir = os.path.join(a.out, "images")
    for v in a.video:
        extract_frames(v, a.n, img_dir)
    run_ui(img_dir, os.path.join(a.out, "labels"), a.scale)


def _self_check():
    one = [(100.0, 50.0), (200.0, 150.0), (100.0, 250.0), (0.0, 150.0)]  # 팁/오른쪽/목/왼쪽
    line = yolo_kpt_line(one, 1000, 1000)
    parts = line.split()
    assert len(parts) == 17
    assert parts[0] == "0"
    cx, cy, bw, bh = (float(v) for v in parts[1:5])
    assert abs(cx - 0.1) < 1e-6 and abs(cy - 0.15) < 1e-6
    assert abs(bw - 0.2) < 1e-6 and abs(bh - 0.2) < 1e-6
    two = [one, [(600.0, 100.0), (700.0, 200.0), (600.0, 300.0), (500.0, 200.0)]]
    lines = yolo_kpt_lines(two, 1000, 1000)
    assert lines.count("\n") == 2  # 선수 2명 -> 라켓 2개 -> 두 줄
    assert yolo_kpt_lines([], 1000, 1000) == ""  # 라켓 없음 -> 빈 라벨

    import tempfile
    d = tempfile.mkdtemp()
    open(os.path.join(d, "f.txt"), "w").write(lines)
    loaded = load_rackets(os.path.join(d, "f.jpg"), d, 1000, 1000)
    assert len(loaded) == 2
    for points, (opoints) in zip(loaded, two):
        for (x, y), (ox, oy) in zip(points, opoints):
            assert max(abs(x - ox), abs(y - oy)) < 1  # 저장/로드 왕복
    print("self-check OK")


if __name__ == "__main__":
    import sys
    _self_check() if len(sys.argv) == 1 else main()
