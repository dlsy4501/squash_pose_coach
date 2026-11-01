"""
스포츠 중계 화면 스타일 오버레이 렌더러 (Pillow).

위쪽 스코어보드(선수별 타격 횟수 + 영상 시간), 타격 순간 배너, 선수 이름표/발밑 링,
왼쪽 아래 이동 거리 패널, 오른쪽 아래 코트맵(궤적 포함)을 영상 위에 얹는다.
cv2.putText는 한글이 안 나와서 Pillow + 시스템 폰트를 쓴다.

render()는 BGR 프레임을 받아 BGR 프레임을 돌려준다. 좌표 단위는 모두 1080p 기준으로 짜고
영상 높이에 비례해 스케일한다.
"""
import math
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

COURT_W, COURT_L, SHORT_LINE, BOX = 6.4, 9.75, 5.44, 1.6  # m

PLAYER_COLORS = [(56, 217, 130), (255, 178, 36)]  # RGB: P1 초록, P2 주황
PANEL = (12, 16, 26, 205)
PANEL_LINE = (255, 255, 255, 38)
TEXT = (244, 247, 252)
MUTED = (150, 162, 184)
COURT_FLOOR = (34, 46, 62, 255)

# 한글 되는 폰트 후보 (굵은 글씨는 stroke로 흉내)
FONTS = [
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/Library/Fonts/AppleGothic.ttf",
    "C:/Windows/Fonts/malgunbd.ttf",
    "C:/Windows/Fonts/malgun.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
]


def _font_path():
    return next((p for p in FONTS if os.path.exists(p)), None)


def fmt_time(t):
    t = int(t)
    return f"{t // 60:02d}:{t % 60:02d}"


def with_alpha(rgb, a):
    return (*rgb, int(max(0, min(255, a))))


def draw_court(d, ox, oy, cw, ch, line=(255, 255, 255, 150), lw=2):
    """스쿼시 코트 바닥 라인 (앞벽이 위쪽). (ox, oy)=왼쪽 위, cw x ch 픽셀."""
    sx, sy = cw / COURT_W, ch / COURT_L
    X = lambda x: ox + x * sx
    Y = lambda y: oy + y * sy
    d.rectangle((ox, oy, ox + cw, oy + ch), fill=COURT_FLOOR, outline=line, width=lw)
    d.line((X(0), Y(SHORT_LINE), X(COURT_W), Y(SHORT_LINE)), fill=line, width=lw)
    d.line((X(COURT_W / 2), Y(SHORT_LINE), X(COURT_W / 2), Y(COURT_L)), fill=line, width=lw)
    for x0, x1 in ((0, BOX), (COURT_W - BOX, COURT_W)):  # 서비스 박스
        d.rectangle((X(x0), Y(SHORT_LINE), X(x1), Y(SHORT_LINE + BOX)), outline=line, width=lw)


def court_diagram(points, width=300):
    """보정할 점들의 위치를 번호로 보여주는 안내 그림(RGB ndarray). points: [(x_m, y_m), ...]"""
    pad, ch = 24, int(width * COURT_L / COURT_W)
    img = Image.new("RGB", (width + 2 * pad, ch + 2 * pad), (20, 24, 34))
    d = ImageDraw.Draw(img, "RGBA")
    draw_court(d, pad, pad, width, ch, line=(255, 255, 255, 200), lw=2)
    f = ImageFont.truetype(_font_path(), 18) if _font_path() else ImageFont.load_default()
    d.text((pad + width / 2, 8), "앞벽", font=f, fill=MUTED, anchor="mt")
    for i, (x, y) in enumerate(points, 1):
        cx, cy = pad + x / COURT_W * width, pad + y / COURT_L * ch
        d.ellipse((cx - 13, cy - 13, cx + 13, cy + 13), fill=(255, 70, 70, 255))
        d.text((cx, cy), str(i), font=f, fill=(255, 255, 255), anchor="mm")
    return np.asarray(img)


class Overlay:
    def __init__(self, height, names, calibrated):
        self.s = height / 1080
        self.names = names
        self.calibrated = calibrated
        path = _font_path()
        mk = lambda px: ImageFont.truetype(path, max(8, int(px * self.s))) if path else ImageFont.load_default()
        self.f_xs, self.f_sm, self.f_md, self.f_lg, self.f_xl = mk(18), mk(23), mk(30), mk(40), mk(56)

    # ---- 작은 도우미 ----
    def _text(self, d, xy, s, font, fill=TEXT, anchor="la", bold=False, alpha=255):
        d.text(xy, s, font=font, fill=with_alpha(fill, alpha), anchor=anchor,
               stroke_width=1 if bold else 0, stroke_fill=with_alpha(fill, alpha))

    def _panel(self, d, box, alpha=1.0, radius=14):
        r = radius * self.s
        d.rounded_rectangle(box, r, fill=with_alpha(PANEL[:3], PANEL[3] * alpha),
                            outline=with_alpha(PANEL_LINE[:3], PANEL_LINE[3] * alpha), width=1)

    # ---- 요소들 ----
    def _player(self, d, p):
        s = self.s
        x1, y1, x2, y2 = p["box"]
        fx, fy = p["foot"]
        col = p["color"]
        rw = max(40 * s, (x2 - x1) * 0.8) / 2
        rh = rw * 0.32
        d.ellipse((fx - rw, fy - rh, fx + rw, fy + rh), fill=with_alpha(col, 45))
        for grow, a, wd in ((1.3, 40, 2), (1.15, 90, 2), (1.0, 235, 4)):
            d.ellipse((fx - rw * grow, fy - rh * grow, fx + rw * grow, fy + rh * grow),
                      outline=with_alpha(col, a), width=max(1, int(wd * s)))

        label = p["name"]
        sub = f'{p["speed"]:.1f} m/s' if self.calibrated else None
        tw = max(d.textlength(label, font=self.f_md), d.textlength(sub or "", font=self.f_xs))
        w_, h_ = tw + 40 * s, (62 if sub else 42) * s
        cx = (x1 + x2) / 2
        bx = cx - w_ / 2
        by = y1 - 16 * s - h_
        bx = max(8 * s, bx)
        by = max(8 * s, by)
        self._panel(d, (bx, by, bx + w_, by + h_), radius=10)
        d.rounded_rectangle((bx, by, bx + 8 * s, by + h_), 4 * s, fill=with_alpha(col, 255))
        self._text(d, (bx + 24 * s, by + 8 * s), label, self.f_md, bold=True)
        if sub:
            self._text(d, (bx + 24 * s, by + 40 * s), sub, self.f_xs, MUTED)
        # 이름표 꼬리
        d.polygon([(cx - 9 * s, by + h_), (cx + 9 * s, by + h_), (cx, by + h_ + 10 * s)],
                  fill=with_alpha(PANEL[:3], PANEL[3]))

    def _scoreboard(self, d, W, st):
        s = self.s
        bw, cw_, bh = 360 * s, 150 * s, 84 * s
        x0 = W / 2 - (2 * bw + cw_) / 2
        y0 = 24 * s
        P = st["players"]
        for side in (0, 1):
            p = P[side] if side < len(P) else None
            col = p["color"] if p else (90, 96, 110)
            name = p["name"] if p else "—"
            shots = p["shots"] if p else 0
            big = st["score"][side] if st.get("score") is not None and p else shots
            small = f"{shots} SHOTS" if st.get("score") is not None else "SHOTS"
            bx = x0 if side == 0 else x0 + bw + cw_
            self._panel(d, (bx, y0, bx + bw, y0 + bh), radius=12)
            sx = bx if side == 0 else bx + bw - 10 * s
            d.rounded_rectangle((sx, y0, sx + 10 * s, y0 + bh), 5 * s, fill=with_alpha(col, 255))
            if side == 0:
                self._text(d, (bx + 30 * s, y0 + 14 * s), name, self.f_md, bold=True)
                self._text(d, (bx + 30 * s, y0 + 52 * s), small, self.f_xs, MUTED)
                self._text(d, (bx + bw - 26 * s, y0 + bh / 2), str(big), self.f_xl, anchor="rm", bold=True)
            else:
                self._text(d, (bx + bw - 30 * s, y0 + 14 * s), name, self.f_md, anchor="ra", bold=True)
                self._text(d, (bx + bw - 30 * s, y0 + 52 * s), small, self.f_xs, MUTED, anchor="ra")
                self._text(d, (bx + 26 * s, y0 + bh / 2), str(big), self.f_xl, anchor="lm", bold=True)
        cx0 = x0 + bw
        d.rectangle((cx0, y0, cx0 + cw_, y0 + bh), fill=with_alpha((6, 8, 14), 235))
        self._text(d, (cx0 + cw_ / 2, y0 + 30 * s), fmt_time(st["t"]), self.f_lg, anchor="mm", bold=True)
        self._text(d, (cx0 + cw_ / 2, y0 + 62 * s), "SCORE" if st.get("score") is not None else "SQUASH",
                   self.f_xs, MUTED, anchor="mm")

    def _banner(self, d, W, b):
        s = self.s
        w_, h_ = 420 * s, 66 * s
        x0, y0 = W / 2 - w_ / 2, 24 * s + 84 * s + 14 * s
        a = b["alpha"]
        self._panel(d, (x0, y0, x0 + w_, y0 + h_), alpha=a, radius=12)
        d.rounded_rectangle((x0, y0, x0 + 10 * s, y0 + h_), 5 * s, fill=with_alpha(b["color"], 255 * a))
        self._text(d, (x0 + 32 * s, y0 + h_ / 2), b["title"], self.f_lg, anchor="lm", bold=True, alpha=255 * a)
        self._text(d, (x0 + w_ - 28 * s, y0 + h_ / 2), b["sub"], self.f_md, b["color"], anchor="rm",
                   bold=True, alpha=255 * a)

    def _distance(self, d, H, P):
        s = self.s
        pw, row = 400 * s, 46 * s
        ph = 62 * s + row * max(1, len(P))
        x0, y0 = 28 * s, H - 28 * s - ph
        self._panel(d, (x0, y0, x0 + pw, y0 + ph))
        self._text(d, (x0 + 22 * s, y0 + 16 * s), "DISTANCE COVERED", self.f_sm, bold=True)
        self._text(d, (x0 + 22 * s, y0 + 40 * s), "코트 위 추정 이동 거리", self.f_xs, MUTED)
        top = max([p["dist"] for p in P] + [1.0])
        for i, p in enumerate(P):
            ry = y0 + 70 * s + i * row
            d.ellipse((x0 + 22 * s, ry + 8 * s, x0 + 36 * s, ry + 22 * s), fill=with_alpha(p["color"], 255))
            self._text(d, (x0 + 48 * s, ry + 4 * s), p["name"], self.f_sm, bold=True)
            self._text(d, (x0 + pw - 22 * s, ry + 4 * s), f'{p["dist"]:.1f} m', self.f_md, anchor="ra", bold=True)
            bx0, bx1, by = x0 + 22 * s, x0 + pw - 22 * s, ry + 34 * s
            d.rounded_rectangle((bx0, by, bx1, by + 5 * s), 3 * s, fill=(255, 255, 255, 30))
            d.rounded_rectangle((bx0, by, bx0 + max(4, (bx1 - bx0) * p["dist"] / top), by + 5 * s), 3 * s,
                                fill=with_alpha(p["color"], 255))

    def _map(self, d, W, H, P):
        s = self.s
        cw = 230 * s
        ch = cw * COURT_L / COURT_W
        pw, ph = cw + 36 * s, ch + 70 * s
        x0, y0 = W - 28 * s - pw, H - 28 * s - ph
        self._panel(d, (x0, y0, x0 + pw, y0 + ph))
        self._text(d, (x0 + 18 * s, y0 + 14 * s), "COURT MAP", self.f_sm, bold=True)
        ox, oy = x0 + 18 * s, y0 + 48 * s
        draw_court(d, ox, oy, cw, ch, lw=max(1, int(2 * s)))
        clip = lambda v, hi: max(-0.2, min(hi + 0.2, v))
        pt = lambda p_: (ox + clip(p_[0], COURT_W) / COURT_W * cw, oy + clip(p_[1], COURT_L) / COURT_L * ch)
        for p in P:
            tr = [pt(q) for q in p["trail"][-120:]]
            n = len(tr)
            for i in range(1, n):
                d.line((tr[i - 1], tr[i]), fill=with_alpha(p["color"], 30 + 170 * i / n), width=max(1, int(3 * s)))
            if tr:
                x, y = tr[-1]
                r = 11 * s
                d.ellipse((x - r, y - r, x + r, y + r), fill=with_alpha(p["color"], 255),
                          outline=(255, 255, 255, 255), width=max(1, int(2 * s)))
                self._text(d, (x, y), p["name"][:1], self.f_xs, (10, 14, 22), anchor="mm", bold=True)

    def _ball(self, d, b):
        """공의 최근 궤적(꼬리가 옅어지는 선)과 현재 위치."""
        s = self.s
        tr = b["trail"]
        for i in range(1, len(tr)):
            d.line((tr[i - 1], tr[i]), fill=(255, 235, 90, int(40 + 190 * i / len(tr))), width=max(1, int(5 * s)))
        if b["now"]:
            x, y = tr[-1]
            for r, a in ((18, 50), (12, 120), (7, 255)):
                d.ellipse((x - r * s, y - r * s, x + r * s, y + r * s), fill=(255, 235, 90, a))

    # ---- 진입점 ----
    def render(self, frame, st):
        """frame: BGR ndarray. st: {t, players:[{name,color,box,foot,dist,shots,speed,trail,visible}], banner}"""
        H, W = frame.shape[:2]
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer, "RGBA")
        for p in st["players"]:
            if p["visible"]:
                self._player(d, p)
        if st.get("ball"):
            self._ball(d, st["ball"])
        self._scoreboard(d, W, st)
        if st.get("banner"):
            self._banner(d, W, st["banner"])
        if self.calibrated and st["players"]:
            self._distance(d, H, st["players"])
            self._map(d, W, H, st["players"])
        base = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
        out = np.asarray(Image.alpha_composite(base, layer).convert("RGB"))
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def _self_check():
    frame = np.full((1080, 1920, 3), 90, np.uint8)
    p = lambda i: dict(name=["민수", "지훈"][i], color=PLAYER_COLORS[i], box=(300 + 800 * i, 300, 460 + 800 * i, 760),
                       foot=(380 + 800 * i, 750), dist=12.3 + i, shots=3 + i, speed=1.5,
                       trail=[(1 + i, 6.0), (1.5 + i, 6.5), (2 + i, 7.0)], visible=True)
    st = dict(t=144.4, players=[p(0), p(1)],
              banner=dict(title="SHOT 4", sub="P1 · 민수", color=PLAYER_COLORS[0], alpha=1.0))
    out = Overlay(1080, ["민수", "지훈"], True).render(frame, st)
    assert out.shape == frame.shape and (out != frame).any()
    assert (out[0:5] == frame[0:5]).all()  # 맨 위 가장자리는 건드리지 않음
    # 점수 + 공 궤적도 에러 없이 그려지고 화면에 흔적이 남아야 함
    st2 = dict(st, score=[3, 1], ball=dict(trail=[(900, 500), (950, 480), (1000, 470)], now=True))
    out2 = Overlay(1080, ["민수", "지훈"], True).render(frame, st2)
    assert (out2[440:520, 880:1020] != out[440:520, 880:1020]).any()
    # 보정 안 한 영상: 코트맵/거리 패널 없이도 에러 없이 그려져야 함
    Overlay(720, ["A", "B"], False).render(cv2.resize(frame, (1280, 720)), dict(st, players=[p(0)]))
    assert court_diagram([(0, 0), (3.2, 5.44)]).shape[2] == 3
    assert fmt_time(144.4) == "02:24"
    print("self-check OK")


if __name__ == "__main__":
    _self_check()
