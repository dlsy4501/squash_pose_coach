"""
사용자 영상 한 프레임의 관절 각도를, 여러 선수 영상 샘플(build_reference.py가 만든
reference.json)과 비교해 어떤 관절이 얼마나 벗어났는지 출력한다.

몸 방향(카메라를 얼마나 정면으로 보고 있는지)이 비슷한 샘플만 걸러서 평균/표준편차를
내기 때문에, 완전히 다른 각도에서 찍은 레퍼런스와 비교되는 걸 어느 정도 막는다.
depth 없이 2D keypoint만으로는 정확한 3D 방향은 알 수 없어서, 어깨 너비/몸통 길이
비율을 방향의 근사치로 쓴다 (정면일수록 어깨가 넓게 보임).

사용법:
    python pose_compare.py --reference reference.json --user user.mp4 --user-frame 30

user-frame은 임팩트 순간 프레임 번호를 직접 지정한다.
자동 임팩트 탐지, 여러 프레임 시계열 정렬(DTW), 다중 카메라는 여기 없음.
"""
import argparse
import math
import statistics

# COCO keypoint indices (YOLO pose 기본 포맷)
KP = {
    "L_SHOULDER": 5, "R_SHOULDER": 6,
    "L_ELBOW": 7, "R_ELBOW": 8,
    "L_WRIST": 9, "R_WRIST": 10,
    "L_HIP": 11, "R_HIP": 12,
    "L_KNEE": 13, "R_KNEE": 14,
    "L_ANKLE": 15, "R_ANKLE": 16,
}

# (관절 이름, 각도를 이루는 세 점) - 가운데 점이 각도의 꼭짓점
ANGLES = {
    "오른팔꿈치": ("R_SHOULDER", "R_ELBOW", "R_WRIST"),
    "왼팔꿈치": ("L_SHOULDER", "L_ELBOW", "L_WRIST"),
    "오른무릎": ("R_HIP", "R_KNEE", "R_ANKLE"),
    "왼무릎": ("L_HIP", "L_KNEE", "L_ANKLE"),
    "오른어깨": ("R_HIP", "R_SHOULDER", "R_ELBOW"),
}

FEEDBACK_THRESHOLD_DEG = 15  # 이 이상 벌어지면 피드백 출력

# ponytail: 경험적으로 정한 임계값. 방향 근사치(orientation_ratio)가 절대적인
# 각도 단위가 아니라서 "몇 도 차이"로 못 정함 - 실제 샘플 모아보고 조정 필요
ORIENTATION_TOLERANCE = 0.15
MIN_ORIENTATION_MATCHES = 2  # 이보다 적게 매칭되면 방향 필터 없이 전체 샘플 사용


def angle(a, b, c):
    """b를 꼭짓점으로 하는 a-b-c 각도(도). a/b/c는 (x, y) 튜플."""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag = math.hypot(*v1) * math.hypot(*v2)
    if mag == 0:
        return None
    cos_theta = max(-1.0, min(1.0, dot / mag))
    return math.degrees(math.acos(cos_theta))


def extract_keypoints(video_path, frame_idx, model):
    import cv2
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"{video_path}의 {frame_idx}번 프레임을 읽을 수 없습니다")

    result = model(frame, verbose=False)[0]
    if result.keypoints is None or len(result.keypoints.xy) == 0:
        raise ValueError(f"{video_path} 프레임 {frame_idx}에서 사람을 찾지 못했습니다")

    xy = result.keypoints.xy[0].tolist()  # 첫 번째로 검출된 사람 기준
    return {name: tuple(xy[idx]) for name, idx in KP.items()}


def compute_angles(kpts):
    out = {}
    for label, (p1, p2, p3) in ANGLES.items():
        out[label] = angle(kpts[p1], kpts[p2], kpts[p3])
    return out


def compute_orientation(kpts):
    """어깨 너비 / 몸통 길이 비율. 정면을 볼수록 크고, 옆을 볼수록 작아짐(요 회전 근사).
    depth가 없어서 정확한 각도는 아니고, "비슷한 방향인지" 비교용 상대값."""
    l_sh, r_sh = kpts["L_SHOULDER"], kpts["R_SHOULDER"]
    l_hip, r_hip = kpts["L_HIP"], kpts["R_HIP"]
    shoulder_width = math.hypot(r_sh[0] - l_sh[0], r_sh[1] - l_sh[1])
    mid_shoulder = ((l_sh[0] + r_sh[0]) / 2, (l_sh[1] + r_sh[1]) / 2)
    mid_hip = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)
    torso_height = math.hypot(mid_shoulder[0] - mid_hip[0], mid_shoulder[1] - mid_hip[1])
    if torso_height == 0:
        return None
    return shoulder_width / torso_height


def select_by_orientation(samples, target, tolerance=ORIENTATION_TOLERANCE):
    """samples 중 target과 방향이 비슷한 것만 반환. 매칭이 너무 적으면 전체로 폴백."""
    if target is None:
        return samples
    matched = [
        s for s in samples
        if s.get("orientation") is not None and abs(s["orientation"] - target) <= tolerance
    ]
    return matched if len(matched) >= MIN_ORIENTATION_MATCHES else samples


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


def compare_to_reference(user_angles, reference):
    """reference[label] = {"mean": ..., "std": ..., "n": ...} (build_reference.py가 생성)"""
    feedback = []
    for label, stat in reference.items():
        u = user_angles.get(label)
        if u is None:
            continue
        diff = u - stat["mean"]
        threshold = max(FEEDBACK_THRESHOLD_DEG, 2 * stat["std"])
        if abs(diff) >= threshold:
            direction = "더 펴야" if diff < 0 else "더 굽혀야"
            feedback.append(
                f"- {label}: 기준 {stat['mean']:.0f}±{stat['std']:.0f}도 vs 나 {u:.0f}도 "
                f"(차이 {abs(diff):.0f}도, 샘플 {stat['n']}개) → {direction} 합니다"
            )
    return feedback


def main():
    import json
    from ultralytics import YOLO

    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, help="build_reference.py로 만든 json")
    parser.add_argument("--user", required=True)
    parser.add_argument("--user-frame", type=int, required=True)
    parser.add_argument("--model", default="yolo11n-pose.pt")
    args = parser.parse_args()

    with open(args.reference, encoding="utf-8") as f:
        samples = json.load(f)  # build_reference.py가 만든 원본 샘플 리스트

    model = YOLO(args.model)
    kpts = extract_keypoints(args.user, args.user_frame, model)
    user_angles = compute_angles(kpts)
    user_orientation = compute_orientation(kpts)

    matched = select_by_orientation(samples, user_orientation)
    if len(matched) < len(samples):
        print(f"[방향 필터] 전체 {len(samples)}개 중 비슷한 방향 {len(matched)}개 샘플로 비교\n")
    reference = aggregate([s["angles"] for s in matched])

    print("관절별 각도 비교 (기준 vs 나)")
    for label, stat in reference.items():
        u = user_angles.get(label)
        print(f"  {label}: {stat['mean']:.0f}±{stat['std']:.0f}도 vs {u:.0f}도" if u else f"  {label}: 검출 실패")

    feedback = compare_to_reference(user_angles, reference)
    print("\n교정 포인트")
    if feedback:
        for line in feedback:
            print(line)
    else:
        print("- 큰 차이 없음")


def _self_check():
    # angle() 자체 검증: 외부 영상/모델 없이도 핵심 로직이 맞는지 확인
    assert abs(angle((1, 0), (0, 0), (0, 1)) - 90) < 1e-6
    assert abs(angle((1, 0), (0, 0), (-1, 0)) - 180) < 1e-6
    assert abs(angle((1, 0), (0, 0), (1, 0)) - 0) < 1e-6

    # compare_to_reference(): 표준편차 안쪽이면 조용히, 크게 벗어나면 피드백
    reference = {"테스트관절": {"mean": 90.0, "std": 2.0, "n": 5}}
    assert compare_to_reference({"테스트관절": 91.0}, reference) == []
    assert len(compare_to_reference({"테스트관절": 120.0}, reference)) == 1

    # compute_orientation(): 정면(어깨 넓게 보임)이 옆모습(어깨 좁게 보임)보다 커야 함
    front = {"L_SHOULDER": (0, 0), "R_SHOULDER": (10, 0), "L_HIP": (1, 10), "R_HIP": (9, 10)}
    side = {"L_SHOULDER": (0, 0), "R_SHOULDER": (2, 0), "L_HIP": (0.5, 10), "R_HIP": (1.5, 10)}
    assert compute_orientation(front) > compute_orientation(side)

    # select_by_orientation(): 비슷한 방향만 골라내고, 매칭 부족하면 전체로 폴백
    samples = [
        {"orientation": 1.0, "angles": {}},
        {"orientation": 1.05, "angles": {}},
        {"orientation": 3.0, "angles": {}},
    ]
    assert len(select_by_orientation(samples, target=1.0, tolerance=0.1)) == 2
    assert select_by_orientation(samples, target=1.0, tolerance=0.001) == samples

    # aggregate(): 평균/표준편차/개수, None은 무시
    ref = aggregate([{"A": 80.0, "B": None}, {"A": 100.0, "B": 50.0}])
    assert ref["A"]["mean"] == 90.0 and ref["A"]["n"] == 2
    assert ref["B"]["n"] == 1

    print("self-check OK")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        _self_check()
    else:
        main()
