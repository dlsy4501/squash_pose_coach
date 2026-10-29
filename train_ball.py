"""
label_ball.py로 만든 ball_dataset으로 공 탐지 모델을 파인튜닝한다.
공 있는 프레임/없는 프레임을 각각 8:2로 나눠 train/val 목록(txt)을 만들고 YOLO 학습을 돌린다.
결과: runs/ball/weights/best.pt

사용법:
    python train_ball.py --epochs 50
공이 작아서(1080p에서 ~15px) imgsz를 크게 쓴다. CUDA GPU가 있으면 자동으로 쓰고, 없으면 CPU(느림).
"""
import argparse
import os
import random


def split(img_dir, lbl_dir, val_ratio=0.2, seed=0):
    """공 있는(라벨 비어있지 않은) 프레임과 없는 프레임을 따로 나눠 비율을 유지한다."""
    imgs = sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg"))
    pos = [f for f in imgs if os.path.getsize(os.path.join(lbl_dir, f[:-4] + ".txt")) > 0]
    neg = [f for f in imgs if f not in set(pos)]
    train, val = [], []
    for group in (pos, neg):
        random.Random(seed).shuffle(group)
        k = max(1, int(len(group) * val_ratio))
        val += group[:k]
        train += group[k:]
    return train, val


def main():
    import torch
    from ultralytics import YOLO
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ball_dataset")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    root = os.path.abspath(a.data)
    img_dir, lbl_dir = os.path.join(root, "images"), os.path.join(root, "labels")
    train, val = split(img_dir, lbl_dir)
    for name, files in (("train", train), ("val", val)):
        with open(os.path.join(root, f"{name}.txt"), "w") as f:
            f.write("\n".join(os.path.join(img_dir, x) for x in files) + "\n")
    with open(os.path.join(root, "data.yaml"), "w") as f:
        f.write(f"path: {root}\ntrain: train.txt\nval: val.txt\nnames:\n  0: ball\n")
    print(f"train {len(train)}장 / val {len(val)}장")

    YOLO("yolo11n.pt").train(
        data=os.path.join(root, "data.yaml"), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
        device=a.device, project=os.path.abspath("runs"), name="ball", exist_ok=True,
        fliplr=0.5, flipud=0.0, mosaic=0.5,  # 공은 작아서 mosaic은 약하게
    )


def _self_check():
    import tempfile
    d = tempfile.mkdtemp()
    os.makedirs(f"{d}/i"); os.makedirs(f"{d}/l")
    for k in range(10):
        open(f"{d}/i/{k}.jpg", "w").close()
        open(f"{d}/l/{k}.txt", "w").write("0 .5 .5 .1 .1\n" if k < 5 else "")
    train, val = split(f"{d}/i", f"{d}/l", 0.2)
    assert len(val) == 2 and len(train) == 8 and set(train).isdisjoint(val)
    pos = {f"{k}.jpg" for k in range(5)}
    assert len(pos & set(val)) == 1  # 공 있는 프레임도 val에 비율대로 들어감
    print("self-check OK")


if __name__ == "__main__":
    import sys
    _self_check() if len(sys.argv) == 1 else main()
