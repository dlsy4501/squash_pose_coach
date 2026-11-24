"""
라켓 위치 라벨링 도구. 영상에서 프레임을 무작위로 뽑아, 라켓을 드래그로 감싸면 YOLO 형식 라벨로 저장한다.
한 프레임에 선수가 둘이면 라켓도 보통 둘이므로, 한 프레임에 박스를 여러 개 그릴 수 있다.
(racket_dataset/images/*.jpg, racket_dataset/labels/*.txt, 클래스 0 = racket, 한 줄에 박스 하나)

공은 크기가 거의 일정해서 클릭 한 번(중심점 + 고정 박스)으로 라벨링했지만, 라켓은 스윙에 따라
화면에서 크기와 방향이 크게 바뀌므로 점이 아니라 드래그로 박스를 직접 그린다.

사용법:
    python label_racket.py --video IMG_0433.MOV IMG_7867.MOV --n 150   # 영상 여러 개 가능
키: 드래그=라켓 박스 추가(한 프레임에 여러 번 가능, 보통 선수 수만큼) / u=마지막 박스 취소
    / n 또는 엔터=이 프레임 완료하고 다음 / s=라켓이 안 보임(빈 라벨) / b=이전 프레임으로 / q=종료
이미 라벨된 프레임은 건너뛰므로 q로 나갔다가 같은 명령으로 이어서 할 수 있다. b로 되돌아간 프레임은
기존 박스를 불러와서 보여주므로 추가/취소 후 n으로 다시 저장하면 된다.
영상에서 프레임을 뽑는 부분(extract_frames)과 라벨 파일 경로 규칙은 label_ball.py와 공유한다.
"""
import argparse
import os

from label_ball import extract_frames, label_path


def yolo_box_line(x1, y1, x2, y2, w, h):
    """드래그 시작/끝 픽셀 좌표(순서 무관) -> YOLO 한 줄(정규화)."""
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
    return f"0 {cx / w:.6f} {cy / h:.6f} {bw / w:.6f} {bh / h:.6f}\n"


def yolo_box_lines(boxes, w, h):
    """[(x1, y1, x2, y2), ...] (픽셀) -> YOLO 여러 줄(정규화). 박스가 없으면 빈 문자열(라켓 없음)."""
    return "".join(yolo_box_line(x1, y1, x2, y2, w, h) for x1, y1, x2, y2 in boxes)


def load_boxes(img_path, lbl_dir, w, h):
    """기존 라벨 파일 -> [(x1, y1, x2, y2), ...] (픽셀). 없으면 빈 리스트."""
    p = label_path(img_path, lbl_dir)
    if not os.path.exists(p):
        return []
    boxes = []
    with open(p) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
            cx, cy, bw, bh = cx * w, cy * h, bw * w, bh * h
            boxes.append((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
    return boxes


def run_ui(img_dir, lbl_dir, scale, min_box):
    import cv2
    os.makedirs(lbl_dir, exist_ok=True)
    imgs = sorted(os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".jpg"))
    todo = [i for i, p in enumerate(imgs) if not os.path.exists(label_path(p, lbl_dir))]
    if not todo:
        print("라벨할 프레임이 없습니다")
        return
    i = todo[0]
    drag = {"start": None, "end": None, "dragging": False}

    def on_mouse(event, mx, my, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            drag["start"], drag["end"], drag["dragging"] = (mx / scale, my / scale), None, True
        elif event == cv2.EVENT_MOUSEMOVE and drag["dragging"]:
            drag["end"] = (mx / scale, my / scale)
        elif event == cv2.EVENT_LBUTTONUP and drag["dragging"]:
            drag["end"], drag["dragging"] = (mx / scale, my / scale), False

    cv2.namedWindow("label_racket")
    cv2.setMouseCallback("label_racket", on_mouse)
    while 0 <= i < len(imgs):
        img = cv2.imread(imgs[i])
        h, w = img.shape[:2]
        boxes = load_boxes(imgs[i], lbl_dir, w, h)
        done = sum(os.path.exists(label_path(p, lbl_dir)) for p in imgs)
        drag["start"] = drag["end"] = None
        drag["dragging"] = False
        next_i = None
        while next_i is None:
            show = cv2.resize(img, None, fx=scale, fy=scale)
            cv2.putText(show, f"{i + 1}/{len(imgs)}  labeled {done}  racket {len(boxes)}"
                               "  drag=add  u=undo  n/enter=next  s=none  b=back  q=quit",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            for x1, y1, x2, y2 in boxes:
                cv2.rectangle(show, (int(x1 * scale), int(y1 * scale)), (int(x2 * scale), int(y2 * scale)),
                              (0, 255, 0), 2)
            if drag["start"] and drag["end"]:
                p1 = (int(drag["start"][0] * scale), int(drag["start"][1] * scale))
                p2 = (int(drag["end"][0] * scale), int(drag["end"][1] * scale))
                cv2.rectangle(show, p1, p2, (0, 200, 255), 2)
            cv2.imshow("label_racket", show)
            key = cv2.waitKey(30) & 0xFF
            if not drag["dragging"] and drag["start"] and drag["end"]:
                x1, y1 = drag["start"]
                x2, y2 = drag["end"]
                if max(abs(x2 - x1), abs(y2 - y1)) >= min_box:
                    boxes.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
                drag["start"] = drag["end"] = None
            if key == ord("u") and boxes:
                boxes.pop()
            elif key == ord("s"):
                boxes = []
                open(label_path(imgs[i], lbl_dir), "w").close()
                next_i = i + 1
            elif key in (ord("n"), 13):
                with open(label_path(imgs[i], lbl_dir), "w") as f:
                    f.write(yolo_box_lines(boxes, w, h))
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
    ap.add_argument("--min-box", type=int, default=8, help="이보다 작은 드래그는 무시(오클릭 방지)")
    ap.add_argument("--scale", type=float, default=0.75, help="화면 표시 배율")
    a = ap.parse_args()
    if not a.video:
        ap.error("--video 가 필요합니다")
    img_dir = os.path.join(a.out, "images")
    for v in a.video:
        extract_frames(v, a.n, img_dir)
    run_ui(img_dir, os.path.join(a.out, "labels"), a.scale, a.min_box)


def _self_check():
    assert yolo_box_line(100, 200, 300, 400, 1000, 1000) == "0 0.200000 0.300000 0.200000 0.200000\n"
    assert yolo_box_line(300, 400, 100, 200, 1000, 1000) == "0 0.200000 0.300000 0.200000 0.200000\n"  # 순서 무관
    two = [(100, 200, 300, 400), (600, 100, 700, 300)]
    lines = yolo_box_lines(two, 1000, 1000)
    assert lines.count("\n") == 2  # 선수 2명 -> 라켓 2개 -> 두 줄
    assert yolo_box_lines([], 1000, 1000) == ""  # 라켓 없음 -> 빈 라벨

    import tempfile
    d = tempfile.mkdtemp()
    open(os.path.join(d, "f.txt"), "w").write(lines)
    loaded = load_boxes(os.path.join(d, "f.jpg"), d, 1000, 1000)
    assert len(loaded) == 2
    for (x1, y1, x2, y2), (ox1, oy1, ox2, oy2) in zip(loaded, two):
        assert max(abs(x1 - ox1), abs(y1 - oy1), abs(x2 - ox2), abs(y2 - oy2)) < 1  # 저장/로드 왕복
    print("self-check OK")


if __name__ == "__main__":
    import sys
    _self_check() if len(sys.argv) == 1 else main()
