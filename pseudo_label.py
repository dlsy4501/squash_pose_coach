"""
1차 공 모델로 새 영상에서 '가짜 라벨' 후보를 만든다 (라벨링 노동을 줄이기 위한 자기 학습).
확신(conf)이 높고 앞뒤 프레임의 궤적과 이어지는 검출만 pseudo_dataset/에 저장한다.
사람이 label_ball.py --review pseudo_dataset 로 O/X만 확인하면 ball_dataset에 합쳐지고, 그걸로 재학습한다.

사용법 (GPU 컴퓨터에서 권장):
    python pseudo_label.py --weights models/ball.pt --video test_video/IMG_7867.MOV test_video/IMG_7868.MOV --n 200
"""
import argparse
import os
import random

from ball_track import pick_track, remove_spikes
from label_ball import yolo_line

WINDOW = 7  # 가운데 프레임을 앞뒤 3프레임과 비교해서 궤적이 이어지는지 본다


def select_confident(window_cands, scale=1.0, min_conf=0.6):
    """연속 프레임 창(홀수 길이)에서 가운데 프레임의 확신 있는 공 한 점 (x, y, conf) 또는 None.
    궤적으로 이어지지 않는 외톨이 검출(그림자, 반사 등)은 버린다."""
    track = remove_spikes(pick_track(window_cands, scale), scale)
    p = track[len(track) // 2]
    if p and p[2] >= min_conf and sum(1 for q in track if q) >= len(track) * 0.6:
        return p
    return None


def main():
    import cv2
    from ultralytics import YOLO
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--video", required=True, nargs="+")
    ap.add_argument("--n", type=int, default=200, help="영상 하나당 살펴볼 구간 수")
    ap.add_argument("--min-conf", type=float, default=0.6)
    ap.add_argument("--box", type=int, default=28)
    ap.add_argument("--out", default="pseudo_dataset")
    a = ap.parse_args()

    model = YOLO(a.weights)
    imgsz = (getattr(model, "ckpt", None) or {}).get("train_args", {}).get("imgsz", 960)
    os.makedirs(f"{a.out}/images", exist_ok=True)
    os.makedirs(f"{a.out}/labels", exist_ok=True)
    for video in a.video:
        stem = os.path.splitext(os.path.basename(video))[0]
        cap = cv2.VideoCapture(video)
        total, h = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        starts = sorted(random.Random(0).sample(range(total - WINDOW), min(a.n, total - WINDOW)))
        kept = 0
        for st in starts:
            cap.set(cv2.CAP_PROP_POS_FRAMES, st)
            frames = [cap.read()[1] for _ in range(WINDOW)]
            if any(f is None for f in frames):
                continue
            cands = []
            for f in frames:
                r = model(f, imgsz=imgsz, conf=0.15, verbose=False)[0]
                cands.append([(float((b[0] + b[2]) / 2), float((b[1] + b[3]) / 2), float(c))
                              for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())])
            p = select_confident(cands, h / 1080, a.min_conf)
            if p:
                name = f"{stem}_f{st + WINDOW // 2:06d}"
                cv2.imwrite(f"{a.out}/images/{name}.jpg", frames[WINDOW // 2], [cv2.IMWRITE_JPEG_QUALITY, 95])
                open(f"{a.out}/labels/{name}.txt", "w").write(
                    yolo_line(p[0], p[1], a.box, frames[0].shape[1], frames[0].shape[0]))
                kept += 1
        cap.release()
        print(f"{stem}: {len(starts)}구간 중 {kept}장 후보 저장")
    print(f"\n다음: python label_ball.py --review {a.out}")


def _self_check():
    line = lambda x: [(x, 400, 0.9)]
    # 쭉 이어지는 궤적의 가운데 -> 선택
    assert select_confident([line(100 + 30 * i) for i in range(7)])[:2] == (190, 400)
    # 가운데만 덩그러니 있는 외톨이 -> 버림
    assert select_confident([[], [], [], [(500, 300, 0.95)], [], [], []]) is None
    # 이어지지만 확신이 낮음 -> 버림
    assert select_confident([[(100 + 30 * i, 400, 0.4)] for i in range(7)]) is None
    print("self-check OK")


if __name__ == "__main__":
    import sys
    _self_check() if len(sys.argv) == 1 else main()
