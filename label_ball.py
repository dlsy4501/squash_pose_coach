"""
공 위치 라벨링 도구. 영상에서 프레임을 무작위로 뽑아, 공을 클릭하면 YOLO 형식 라벨로 저장한다.
(ball_dataset/images/*.jpg, ball_dataset/labels/*.txt, 클래스 0 = ball)

사용법:
    python label_ball.py --video IMG_0433.MOV --n 300
키: 클릭=공 위치 저장 후 다음 / s=공이 안 보임(빈 라벨, 배경 학습용) / b=이전 프레임으로 / q=종료
이미 라벨된 프레임은 건너뛰므로 q로 나갔다가 같은 명령으로 이어서 할 수 있다.
"""
import argparse
import os
import random


def yolo_line(x, y, box, w, h):
    """픽셀 중심(x, y)과 정사각형 박스 크기 -> YOLO 한 줄(정규화)."""
    return f"0 {x / w:.6f} {y / h:.6f} {box / w:.6f} {box / h:.6f}\n"


def extract_frames(video, n, out_dir, seed=0):
    """영상을 처음부터 한 번만 훑으며 뽑는다(임의 seek은 프레임당 ~1초라 느림)."""
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    want = set(random.Random(seed).sample(range(total), min(n, total)))
    print(f"프레임 {len(want)}장 추출 중 (1~2분 걸립니다)...")
    for i in range(total):
        if i not in want:
            cap.grab()
            continue
        ok, frame = cap.read()
        path = os.path.join(out_dir, f"f{i:06d}.jpg")
        if ok and not os.path.exists(path):
            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if i % 2000 == 0:
            print(f"  {i}/{total}")
    cap.release()


def label_path(img_path, lbl_dir):
    return os.path.join(lbl_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt")


def run_ui(img_dir, lbl_dir, box, scale):
    import cv2
    os.makedirs(lbl_dir, exist_ok=True)
    imgs = sorted(os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".jpg"))
    todo = [i for i, p in enumerate(imgs) if not os.path.exists(label_path(p, lbl_dir))]
    if not todo:
        print("라벨할 프레임이 없습니다")
        return
    i = todo[0]
    click = []

    def on_mouse(event, mx, my, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            click.append((mx / scale, my / scale))

    cv2.namedWindow("label_ball")
    cv2.setMouseCallback("label_ball", on_mouse)
    while 0 <= i < len(imgs):
        img = cv2.imread(imgs[i])
        h, w = img.shape[:2]
        show = cv2.resize(img, None, fx=scale, fy=scale)
        done = sum(os.path.exists(label_path(p, lbl_dir)) for p in imgs)
        cv2.putText(show, f"{i + 1}/{len(imgs)}  labeled {done}   click=ball  s=none  b=back  q=quit",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("label_ball", show)
        click.clear()
        while True:
            key = cv2.waitKey(30) & 0xFF
            if click:
                x, y = click[-1]
                with open(label_path(imgs[i], lbl_dir), "w") as f:
                    f.write(yolo_line(x, y, box, w, h))
                i += 1
                break
            if key == ord("s"):
                open(label_path(imgs[i], lbl_dir), "w").close()
                i += 1
                break
            if key == ord("b"):
                i = max(0, i - 1)
                break
            if key == ord("q"):
                cv2.destroyAllWindows()
                return
    cv2.destroyAllWindows()
    print("모든 프레임 라벨 완료")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="ball_dataset")
    ap.add_argument("--box", type=int, default=28, help="라벨 박스 한 변(px). 공이 번진 길이보다 약간 크게")
    ap.add_argument("--scale", type=float, default=0.75, help="화면 표시 배율")
    a = ap.parse_args()
    img_dir = os.path.join(a.out, "images")
    extract_frames(a.video, a.n, img_dir)
    run_ui(img_dir, os.path.join(a.out, "labels"), a.box, a.scale)


def _self_check():
    assert yolo_line(960, 540, 20, 1920, 1080) == "0 0.500000 0.500000 0.010417 0.018519\n"
    assert label_path("a/b/f000012.jpg", "lbl") == os.path.join("lbl", "f000012.txt")
    print("self-check OK")


if __name__ == "__main__":
    import sys
    _self_check() if len(sys.argv) == 1 else main()
