"""
label_racket.py로 만든 racket_dataset으로 라켓 탐지 모델을 파인튜닝한다.
라켓 있는 프레임/없는 프레임을 각각 8:2로 나눠 train/val 목록(txt)을 만들고 YOLO 학습을 돌린다.
결과: runs/racket/weights/best.pt

사용법:
    python train_racket.py --epochs 50
공과 달리 라켓은 화면에서 크게 보이므로 imgsz는 공 모델보다 작아도 된다. CUDA GPU가 있으면 자동으로
쓰고, 없으면 CPU(느림). train/val 분할 로직은 train_ball.py와 공유한다.
"""
import argparse
import os

from train_ball import split


def main():
    import torch
    from ultralytics import YOLO
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="racket_dataset")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    root = os.path.abspath(a.data)
    img_dir, lbl_dir = os.path.join(root, "images"), os.path.join(root, "labels")
    train, val = split(img_dir, lbl_dir)
    for name, files in (("train", train), ("val", val)):
        with open(os.path.join(root, f"{name}.txt"), "w") as f:
            f.write("\n".join(os.path.join(img_dir, x) for x in files) + "\n")
    with open(os.path.join(root, "data.yaml"), "w") as f:
        f.write(f"path: {root}\ntrain: train.txt\nval: val.txt\nnames:\n  0: racket\n")
    print(f"train {len(train)}장 / val {len(val)}장")

    YOLO("yolo11n.pt").train(
        data=os.path.join(root, "data.yaml"), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
        device=a.device, project=os.path.abspath("runs"), name="racket", exist_ok=True,
        fliplr=0.5, flipud=0.0, mosaic=1.0,  # 라켓은 커서 공 모델보다 mosaic을 세게 써도 됨
        workers=a.workers,
    )


def _self_check():
    from train_ball import _self_check as ball_self_check
    ball_self_check()  # split()은 train_ball.py 것을 그대로 재사용하므로 거기서 검증됨


if __name__ == "__main__":
    import sys
    _self_check() if len(sys.argv) == 1 else main()
