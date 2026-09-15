"""
여러 선수 영상(여러 프레임)에서 관절 각도를 뽑아 평균/표준편차로 "이상적인 자세"
레퍼런스를 만든다. pose_compare.py가 이 결과(reference.json)를 사용자 자세와 비교한다.

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
import statistics

from pose_compare import compute_angles, extract_keypoints


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


def aggregate(angle_dicts):
    """[{관절: 각도}, ...] 리스트 -> {관절: {mean, std, n}}"""
    samples = {}
    for angles in angle_dicts:
        for label, val in angles.items():
            if val is not None:
                samples.setdefault(label, []).append(val)

    reference = {}
    for label, vals in samples.items():
        if not vals:
            continue
        reference[label] = {
            "mean": statistics.mean(vals),
            "std": statistics.pstdev(vals),
            "n": len(vals),
        }
    return reference


def build(pairs, model_name):
    from ultralytics import YOLO

    model = YOLO(model_name)
    angle_dicts = [compute_angles(extract_keypoints(video, frame, model)) for video, frame in pairs]
    return aggregate(angle_dicts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", required=True, help="영상,프레임 목록 텍스트 파일")
    parser.add_argument("--out", default="reference.json")
    parser.add_argument("--model", default="yolo11n-pose.pt")
    args = parser.parse_args()

    pairs = parse_list(args.list)
    reference = build(pairs, args.model)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(reference, f, ensure_ascii=False, indent=2)

    print(f"샘플 {len(pairs)}개로 레퍼런스 저장: {args.out}")
    for label, stat in reference.items():
        print(f"  {label}: 평균 {stat['mean']:.1f}도, 표준편차 {stat['std']:.1f} (n={stat['n']})")


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

    # aggregate: 평균/표준편차/개수가 맞는지, None은 무시하는지
    ref = aggregate([{"A": 80.0, "B": None}, {"A": 100.0, "B": 50.0}])
    assert ref["A"]["mean"] == 90.0 and ref["A"]["n"] == 2
    assert ref["B"]["n"] == 1
    assert "C" not in ref  # 값이 없는 관절은 결과에서 빠짐(C는 애초에 없음)

    print("self-check OK")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        _self_check()
    else:
        main()
