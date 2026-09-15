"""
MVP: 레퍼런스(선수) 영상 한 프레임 vs 사용자 영상 한 프레임의 관절 각도를 비교해
어떤 관절이 얼마나 벗어났는지 출력한다.

사용법:
    python pose_compare.py --ref ref.mp4 --ref-frame 42 --user user.mp4 --user-frame 30

ref-frame/user-frame은 임팩트 순간 프레임 번호를 직접 지정한다.
자동 임팩트 탐지, 여러 프레임 시계열 정렬(DTW), 다중 카메라는 여기 없음.
"""
import argparse
import math

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


def compare(ref_angles, user_angles):
    feedback = []
    for label in ANGLES:
        r, u = ref_angles[label], user_angles[label]
        if r is None or u is None:
            continue
        diff = u - r
        if abs(diff) >= FEEDBACK_THRESHOLD_DEG:
            direction = "더 펴야" if diff < 0 else "더 굽혀야"
            feedback.append(
                f"- {label}: 선수 {r:.0f}도 vs 나 {u:.0f}도 (차이 {abs(diff):.0f}도) → {direction} 합니다"
            )
    return feedback


def main():
    from ultralytics import YOLO

    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True)
    parser.add_argument("--ref-frame", type=int, required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--user-frame", type=int, required=True)
    parser.add_argument("--model", default="yolo11n-pose.pt")
    args = parser.parse_args()

    model = YOLO(args.model)
    ref_angles = compute_angles(extract_keypoints(args.ref, args.ref_frame, model))
    user_angles = compute_angles(extract_keypoints(args.user, args.user_frame, model))

    print("관절별 각도 비교 (선수 vs 나)")
    for label in ANGLES:
        r, u = ref_angles[label], user_angles[label]
        print(f"  {label}: {r:.0f}도 vs {u:.0f}도" if r and u else f"  {label}: 검출 실패")

    feedback = compare(ref_angles, user_angles)
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
    print("self-check OK")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        _self_check()
    else:
        main()
