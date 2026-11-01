"""
스쿼시 영상 분석 앱 (웹 UI). 영상을 올리고 → 코트 위 점 몇 개를 클릭하고 → 시작 버튼만 누르면
중계 화면 스타일 오버레이가 입혀진 영상이 나온다. 코드/명령어를 몰라도 쓸 수 있게 하는 것이 목적.

실행: start.command(맥) / start.bat(윈도우)를 더블클릭. 개발자는 `python app.py`.
"""
import json
import os
import time

import cv2
import gradio as gr

import rally
from overlay_ui import PLAYER_COLORS, court_diagram
from track_overlay import CALIB_POINTS, homography_from_pairs, postprocess, process_video, render

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")
BALL_MODEL = os.path.join(HERE, "models", "ball.pt")  # GPU에서 학습한 공 탐지 모델을 여기에 두면 자동으로 쓴다
os.chdir(HERE)  # 모델 가중치(yolo11n-pose.pt)가 이 폴더에 받아지도록

LABELS = [f"{i + 1}. {name}" for i, (name, _) in enumerate(CALIB_POINTS)]
MIN_POINTS = 4
SEC_PER_FRAME = 0.3  # 처리 시간 안내용 대략치 (CPU 기준)


def video_info(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, n / fps


def grab_frame(path, t):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise gr.Error("이 시점의 화면을 읽을 수 없습니다. 다른 시간을 골라 주세요.")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def draw_points(img, pts):
    """클릭한 점을 번호와 함께 화면에 표시 (잘 찍혔는지 눈으로 확인용)."""
    out = img.copy()
    r = max(8, img.shape[0] // 90)
    for i, (x, y) in pts.items():
        cv2.circle(out, (int(x), int(y)), r, (255, 60, 60), 2)
        cv2.circle(out, (int(x), int(y)), 2, (255, 60, 60), -1)
        cv2.putText(out, str(i + 1), (int(x) + r + 4, int(y) - r), cv2.FONT_HERSHEY_SIMPLEX,
                    r / 10, (255, 60, 60), 2)
    return out


def calib_status(pts):
    n = len(pts)
    if n >= MIN_POINTS:
        return f"✅ 점 {n}개 찍음 — 이동 거리와 코트맵이 표시됩니다. (점이 더 많을수록 정확해요)"
    return f"점 {n}개 찍음 — **{MIN_POINTS}개 이상** 찍으면 이동 거리와 코트맵이 표시됩니다. (건너뛰어도 분석은 됩니다)"


def next_label(pts, cur):
    """다음으로 찍을 점: 현재 다음 번호부터 아직 안 찍은 첫 점."""
    order = list(range(cur + 1, len(LABELS))) + list(range(cur + 1))
    return LABELS[next((i for i in order if i not in pts), cur)]


def on_upload(path):
    if not path:
        return gr.update(), None, {}, gr.update(value=LABELS[0]), "", calib_status({})
    fps, dur = video_info(path)
    t = min(5.0, dur / 2)
    est = dur * fps * SEC_PER_FRAME / 60
    info = (f"영상 길이 **{int(dur // 60)}분 {int(dur % 60)}초** ({fps:.0f}fps). "
            f"전체를 분석하면 이 컴퓨터에서 **약 {est:.0f}분** 걸릴 것 같아요 "
            f"(그래픽카드가 없으면 오래 걸립니다. 아래에서 구간을 줄일 수 있어요).")
    return (gr.update(maximum=max(1, int(dur)), value=int(t)), draw_points(grab_frame(path, t), {}), {},
            gr.update(value=LABELS[0]), info, calib_status({}))


def on_seek(path, t, pts):
    if not path:
        return None
    return draw_points(grab_frame(path, t), pts)


def on_click(evt: gr.SelectData, path, t, pts, choice):
    if not path:
        raise gr.Error("먼저 영상을 올려 주세요.")
    cur = LABELS.index(choice)
    pts = dict(pts)
    pts[cur] = tuple(evt.index)
    return (draw_points(grab_frame(path, t), pts), pts, gr.update(value=next_label(pts, cur)),
            calib_status(pts))


def on_skip(pts, choice):
    return gr.update(value=next_label(pts, LABELS.index(choice)))


def on_reset(path, t):
    return (draw_points(grab_frame(path, t), {}) if path else None), {}, gr.update(value=LABELS[0]), calib_status({})


def summary_md(summary, ball_used):
    rows = ["| 선수 | 득점 | 타격 횟수(추정) | 이동 거리 |", "|---|---|---|---|"]
    for s in summary:
        d = f'{s["dist"]:.1f} m' if s["dist"] is not None else "— (코트 보정 안 함)"
        rows.append(f'| {s["name"]} | {s["points"]} | {s["shots"]}회 | {d} |')
    note = ("공 모델을 사용해 타격을 보정했어요." if ball_used else
            "공 모델이 없어서 손목 움직임만으로 타격을 셌어요 (연습 스윙이 섞일 수 있어요).")
    return ("\n".join(rows) + f"\n\n{note} **득점자는 '마지막에 친 사람'으로 추정한 값**이라 틀릴 수 있어요. "
            "아래 표에서 바로잡고 영상을 다시 만들 수 있습니다.")


def rally_rows(data, names):
    """표시용 행: [랠리, 시작(초), 타격 수, 득점자 이름]"""
    who = lambda k: names[k] if k is not None else "무효"
    return [[r["id"], round(data["start"] + r["start"] / data["fps"], 1), len(r["hits"]), who(r["winner"])]
            for r in data["rallies"]]


def parse_overrides(rows, names):
    """표에서 사용자가 고친 득점자 이름 -> {랠리 id: 슬롯 또는 None}. 알아볼 수 없는 값은 무시."""
    out = {}
    for row in (rows.values.tolist() if hasattr(rows, "values") else rows or []):
        rid, who = int(row[0]), str(row[3]).strip()
        if who in ("무효", "-", ""):
            out[rid] = None
        else:
            for k, n in enumerate(names):
                if who in (n, str(k + 1), f"P{k + 1}"):
                    out[rid] = k
    return out


def _names(n1, n2):
    return [(n1 or "P1").strip() or "P1", (n2 or "P2").strip() or "P2"]


def run(path, pts, name1, name2, start, dur):
    if not path:
        raise gr.Error("먼저 영상을 올려 주세요.")
    H = None
    if len(pts) >= MIN_POINTS:
        ids = sorted(pts)
        H = homography_from_pairs([pts[i] for i in ids], [CALIB_POINTS[i][1] for i in ids])
    os.makedirs(OUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    stamp = f"{base}_{time.strftime('%m%d_%H%M%S')}"
    out = os.path.join(OUT_DIR, f"분석_{stamp}.mp4")
    ball = BALL_MODEL if os.path.exists(BALL_MODEL) else None
    names = _names(name1, name2)
    t0, last = time.time(), None
    yield "모델을 준비하는 중... (처음 한 번은 모델을 내려받느라 조금 걸려요)", None, None, "", None, None, ""
    for ev in process_video(path, out, H, float(start or 0), float(dur) if dur else None, names, ball):
        if ev.get("done"):
            data = ev["analysis"]
            apath = os.path.join(OUT_DIR, f"분석_{stamp}.json")
            json.dump(data, open(apath, "w", encoding="utf-8"))
            rally.save_csv(data["rallies"], data["fps"], os.path.join(OUT_DIR, f"분석_{stamp}_랠리.csv"),
                          float(start or 0), names)
            yield (f"✅ 완료! ({(time.time() - t0) / 60:.1f}분 걸림) 영상과 랠리 기록(csv)은 `outputs` 폴더에도 저장돼 있어요.",
                   last, ev["out"], summary_md(ev["summary"], ball is not None), ev["out"],
                   rally_rows(data, names), apath)
            return
        p = ev["progress"]
        eta = (time.time() - t0) / p * (1 - p) / 60 if p > 0.02 else None
        last = cv2.cvtColor(ev["frame"], cv2.COLOR_BGR2RGB)
        yield (f"{ev['stage']} 중... **{p * 100:.0f}%**" + (f" · 남은 시간 약 {eta:.0f}분" if eta is not None else ""),
               last, None, "", None, None, "")


def rerender(apath, rows, name1, name2):
    """표에서 정정한 득점자를 반영해 영상만 다시 만든다 (분석은 다시 안 함)."""
    if not apath:
        raise gr.Error("먼저 분석을 한 번 실행해 주세요.")
    names = _names(name1, name2)
    data = json.load(open(apath, encoding="utf-8"))
    postprocess(data, parse_overrides(rows, names))
    stamp = time.strftime("%H%M%S")
    out = apath.replace(".json", f"_정정{stamp}.mp4")
    rally.save_csv(data["rallies"], data["fps"], apath.replace(".json", f"_정정{stamp}_랠리.csv"),
                  data["start"], names)
    last = None
    for ev in render(data, out, names):
        if ev.get("done"):
            yield ("✅ 점수를 반영해 다시 만들었어요.", last, ev["out"],
                   summary_md(ev["summary"], data["has_ball"]), ev["out"], rally_rows(data, names))
            return
        last = cv2.cvtColor(ev["frame"], cv2.COLOR_BGR2RGB)
        yield f"영상 다시 만드는 중... **{ev['progress'] * 100:.0f}%**", last, None, "", None, gr.update()


def build():
    with gr.Blocks(title="스쿼시 영상 분석", theme=gr.themes.Soft(primary_hue="emerald")) as ui:
        gr.Markdown("# 🎾 스쿼시 영상 분석\n영상을 올리면 선수 이동, 타격, 점수를 중계 화면처럼 입혀 드려요.")
        path = gr.File(label="① 영상 파일을 올려 주세요 (mp4, mov 등)", file_types=["video"], type="filepath")
        info = gr.Markdown()
        pts = gr.State({})
        apath = gr.State("")

        gr.Markdown("### ② 코트 위치 알려주기 (선택)\n"
                    "화면에서 **오른쪽 안내 그림의 번호에 해당하는 점**을 클릭해 주세요. 코트의 바닥 선 위치를 알려주면 "
                    "이동 거리와 코트맵을 계산할 수 있어요. **카메라가 고정된 영상**에서만 정확합니다. "
                    "건너뛰어도 되지만 그러면 이동 거리/코트맵은 나오지 않아요.")
        with gr.Row():
            with gr.Column(scale=3):
                frame = gr.Image(label="클릭해서 점을 찍으세요", interactive=False, type="numpy")
                t = gr.Slider(0, 1, value=0, step=1, label="점이 잘 보이는 장면으로 이동 (초)")
            with gr.Column(scale=1):
                gr.Image(court_diagram([p for _, p in CALIB_POINTS]), label="안내 그림 (코트를 위에서 본 모습)",
                         interactive=False, show_download_button=False)
                choice = gr.Radio(LABELS, value=LABELS[0], label="지금 찍을 점")
                with gr.Row():
                    skip = gr.Button("이 점 건너뛰기", size="sm")
                    reset = gr.Button("모두 지우기", size="sm")
                status = gr.Markdown(calib_status({}))

        gr.Markdown("### ③ 선수 이름과 분석 구간")
        with gr.Row():
            n1 = gr.Textbox(label="선수 1 이름 (초록)", value="P1")
            n2 = gr.Textbox(label="선수 2 이름 (주황)", value="P2")
            start = gr.Number(label="시작 (초)", value=0, precision=0)
            dur = gr.Number(label="길이 (초, 0 = 끝까지)", value=30, precision=0)
        go = gr.Button("🚀 분석 시작", variant="primary", size="lg")

        state_md = gr.Markdown()
        with gr.Row():
            preview = gr.Image(label="진행 화면", interactive=False)
            result = gr.Video(label="결과 영상", interactive=False)
        table = gr.Markdown()
        gr.Markdown("### ④ 랠리별 득점자 확인\n득점자 칸을 **선수 이름**(또는 `무효`)으로 고친 뒤 아래 버튼을 누르면 "
                    "점수가 반영된 영상을 다시 만들어요. (분석은 다시 하지 않아서 빨라요)")
        rallies = gr.Dataframe(headers=["랠리", "시작(초)", "타격 수", "득점자"], datatype=["number", "number", "number", "str"],
                               col_count=(4, "fixed"), interactive=True, label="랠리 목록")
        redo = gr.Button("점수 반영해서 영상 다시 만들기")
        download = gr.File(label="결과 영상 다운로드")

        path.change(on_upload, path, [t, frame, pts, choice, info, status])
        t.release(on_seek, [path, t, pts], frame)
        frame.select(on_click, [path, t, pts, choice], [frame, pts, choice, status])
        skip.click(on_skip, [pts, choice], choice)
        reset.click(on_reset, [path, t], [frame, pts, choice, status])
        go.click(run, [path, pts, n1, n2, start, dur], [state_md, preview, result, table, download, rallies, apath])
        redo.click(rerender, [apath, rallies, n1, n2], [state_md, preview, result, table, download, rallies])
    return ui


if __name__ == "__main__":
    build().queue().launch(inbrowser=True, allowed_paths=[OUT_DIR])
