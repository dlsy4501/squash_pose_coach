"""
여러 선수 영상(여러 프레임)에서 관절 각도 + 몸 방향(orientation)을 뽑아
reference.json에 원본 샘플 리스트로 저장한다.

평균/표준편차는 여기서 미리 내지 않는다 - pose_compare.py가 비교 시점에
사용자와 방향이 비슷한 샘플만 걸러서 그때 집계한다 (compute_orientation 참고).

사용법:
    python build_reference.py --list samples.txt --out reference.json

samples.txt 형식 (한 줄에 하나, 임팩트 순간 프레임 번호를 직접 지정):
    player1.mp4,42
    player1.mp4,110
    player2.mp4,58
    # 주석은 #으로 시작
"""
import argparse
import json

from pose_compare import compute_angles, compute_orientation, extract_keypoints


def parse_list(path):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            video, frame = line.rsplit(",", 1)
            pairs.append((video.strip(), int(frame.strip())))
    return pairs


def build(pairs, model_name):
    from ultralytics import YOLO

    model = YOLO(model_name)
    samples = []
    for video, frame in pairs:
        kpts = extract_keypoints(video, frame, model)
        samples.append({"orientation": compute_orientation(kpts), "angles": compute_angles(kpts)})
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", required=True, help="영상,프레임 목록 텍스트 파일")
    parser.add_argument("--out", default="reference.json")
    parser.add_argument("--model", default="yolo11n-pose.pt")
    args = parser.parse_args()

    pairs = parse_list(args.list)
    samples = build(pairs, args.model)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"샘플 {len(samples)}개 저장: {args.out}")


def _self_check():
    # parse_list: 주석/빈 줄 무시하고 (video, frame) 튜플로 파싱되는지
    import tempfile, os
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("# comment\n\na.mp4,10\nb.mp4, 20 \n")
        path = f.name
    try:
        assert parse_list(path) == [("a.mp4", 10), ("b.mp4", 20)]
    finally:
        os.remove(path)

    print("self-check OK")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        _self_check()
    else:
        main()
