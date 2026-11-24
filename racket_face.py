"""
라켓 헤드 4점(label_racket.py가 찍은 팁/오른쪽/목/왼쪽, 또는 racket-pose 모델이 예측한 같은 순서의
키포인트)으로 "라켓 면이 카메라 쪽으로 얼마나 열려 있는지"를 추정한다.

원리: 실제 라켓 헤드는 거의 원(타원)이다. 면이 카메라를 정면으로 보고 있으면(= 가장 열린 상태) 화면에는
거의 원래 비율의 타원으로 보이고, 면이 카메라와 수직이 되도록 돌아가면(= 닫힌 상태, 테두리만 옆에서 보임)
한쪽 축이 짧아져 찌그러진 타원~직선에 가깝게 보인다.
팁-목을 긴 축(길이 방향), 오른쪽-왼쪽을 짧은 축(너비 방향)으로 보면, 긴 축은 카메라 쪽으로 기울어도
화면에 보이는 길이가 거의 그대로지만(화면 평면에서 회전만 일어남) 너비 쪽은 cos(theta)만큼 줄어든다
(theta = 면이 카메라 정면에서 돌아간 각도). 그래서

    ratio = 너비(오른쪽-왼쪽 거리) / 길이(팁-목 거리)
    theta = arccos(clip(ratio / base_ratio, 0, 1))   # base_ratio: 면이 완전히 열렸을 때(정면)의 너비/길이

base_ratio는 실제 라켓 헤드 모양(스쿼시 라켓은 거의 원형~타원형)에서 보통 0.8~1.0 정도이며,
라벨링된 "면이 거의 정면인" 프레임들의 ratio 분포로 보정하는 게 가장 정확하다. 기본값은 0.9로 둔다.

theta = 0도  -> 면이 카메라 쪽으로 완전히 열려 있음 (헤드가 거의 원래 비율로 보임)
theta = 90도 -> 면이 카메라와 수직(거의 안 보임, 가장 닫힌 상태)
"""
import math


def ellipse_axes(points):
    """4점(팁, 오른쪽, 목, 왼쪽) -> (길이축 거리, 너비축 거리)."""
    tip, right, throat, left = points
    length = math.hypot(tip[0] - throat[0], tip[1] - throat[1])
    width = math.hypot(right[0] - left[0], right[1] - left[1])
    return length, width


def openness(points, base_ratio=0.9):
    """4점 -> (ratio, theta_degrees). length가 0이면 (0.0, 90.0)."""
    length, width = ellipse_axes(points)
    if length <= 0:
        return 0.0, 90.0
    ratio = width / length
    theta = math.degrees(math.acos(max(0.0, min(1.0, ratio / base_ratio))))
    return ratio, theta


def _self_check():
    square = [(0, -10), (10, 0), (0, 10), (-10, 0)]  # 길이 20, 너비 20 -> 정면(ratio=1)
    ratio, theta = openness(square, base_ratio=1.0)
    assert abs(ratio - 1.0) < 1e-9 and abs(theta - 0.0) < 1e-9

    thin = [(0, -10), (1, 0), (0, 10), (-1, 0)]  # 너비가 아주 작음 -> 거의 닫힘
    ratio, theta = openness(thin, base_ratio=1.0)
    assert ratio < 0.15 and theta > 80

    same_point = [(0, 0), (0, 0), (0, 0), (0, 0)]
    assert openness(same_point) == (0.0, 90.0)
    print("self-check OK")


if __name__ == "__main__":
    _self_check()
