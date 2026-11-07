#!/usr/bin/env python3
"""Cut the recorded shots into the README demo with ffmpeg.

    python3 scripts/demo/edit.py [--out docs/media]

Reads $DEMO/rec/<shot>.webm and <shot>.json from scripts/demo/record.mjs. Waits (model work the
viewer cannot see) are sped up; streamed text and the measured tok/s are left at real speed.
Writes demo.mp4, demo.webp and demo-poster.png.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

DEMO = Path(os.environ.get("DEMO", "/tmp/arc-demo"))
REC = DEMO / "rec"
WORK = DEMO / "edit"
W, H, FPS = 1440, 900, 30
FADE = 0.3  # between shots and between cuts inside a shot
FONT = "Inter"
BAND, BAND_COLOR = 64, "0x0e0e16"  # caption strip under the page


def run(*args: str) -> None:
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *args], check=True)


def duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         check=True, capture_output=True, text=True).stdout
    return float(out)


def zoom(box: dict, scale_h: float) -> str:
    """A 16:10 crop around ``box`` (page pixels), kept inside the frame."""
    h = min(H, max(360, round(box["h"] * scale_h)))
    w = round(h * W / H)
    cx, cy = box["x"] + box["w"] / 2, box["y"] + box["h"] / 2
    x = min(max(0, round(cx - w / 2)), W - w)
    y = min(max(0, round(cy - h / 2)), H - h)
    return f"crop={w}:{h}:{x}:{y},scale={W}:{H}:flags=lanczos"


def piece(src: Path, out: Path, start: float, end: float, speed: float = 1.0, crop: str | None = None) -> Path:
    vf = [f"setpts=(PTS-STARTPTS)/{speed}"]
    if crop:
        vf.append(crop)
    vf += [f"fps={FPS}", f"scale={W}:{H}", "format=yuv420p"]
    run("-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(src), "-an", "-vf", ",".join(vf),
        "-c:v", "libx264", "-crf", "12", "-preset", "veryfast", str(out))
    return out


def join(parts: list[tuple[Path, str]], out: Path) -> Path:
    """Join pieces in order; each piece after the first says how it meets the one before: cut or fade."""
    inputs, chain, length = [], "", 0.0
    for i, (path, _) in enumerate(parts):
        inputs += ["-i", str(path)]
        # concat and xfade need matching timebases and frame rates.
        chain += f"[{i}:v]fps={FPS},setpts=PTS-STARTPTS,settb=1/{FPS}[n{i}];"
    for i, (path, how) in enumerate(parts):
        d = duration(path)
        if i == 0:
            label, length = "[n0]", d
            continue
        nxt = f"[j{i}]"
        if how == "fade":
            chain += f"{label}[n{i}]xfade=transition=fade:duration={FADE}:offset={length - FADE:.3f},settb=1/{FPS}{nxt};"
            length += d - FADE
        else:
            chain += f"{label}[n{i}]concat=n=2:v=1:a=0,settb=1/{FPS}{nxt};"
            length += d
        label = nxt
    if len(parts) == 1:
        run(*inputs, "-c", "copy", str(out))
    else:
        run(*inputs, "-filter_complex", chain.rstrip(";"), "-map", label, "-c:v", "libx264", "-crf", "12",
            "-preset", "veryfast", "-pix_fmt", "yuv420p", str(out))
    return out


def caption(src: Path, text: str, out: Path) -> Path:
    d = duration(src)
    alpha = f"if(lt(t,0.35),t/0.35,if(gt(t,{d - 0.45:.3f}),max(0,({d - 0.1:.3f}-t)/0.35),1))"
    # The UI fills every corner of the 1440x900 page, so the caption sits on a strip below it
    # in the app's background colour instead of covering buttons.
    draw = (f"pad={W}:{H + BAND}:0:0:color={BAND_COLOR},"
            f"drawtext=font='{FONT}':text='{text}':fontsize=28:fontcolor=0xe8eaf0:alpha='{alpha}'"
            f":x=36:y={H}+({BAND}-th)/2")
    run("-i", str(src), "-vf", draw, "-c:v", "libx264", "-crf", "12", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(out))
    return out


def shots() -> tuple[list[Path], float]:
    m = {name: json.loads((REC / f"{name}.json").read_text()) for name in ("terminal", "discover", "chat", "work")}
    out = []

    # 1. `arc-llama start`, then the page it opened (the blank page load in between is cut).
    t, src = m["terminal"], REC / "terminal.webm"
    a = piece(src, WORK / "1a.mp4", 0.5, t["browser"])
    b = piece(src, WORK / "1b.mp4", t["page"], t["page"] + 3.0)
    out.append(caption(join([(a, ""), (b, "fade")], WORK / "1.mp4"), "Detects your Arc GPU", WORK / "1c.mp4"))

    # 2. Discover: ease down to the shelves, then push in on the hovered card.
    t, src = m["discover"], REC / "discover.webm"
    a = piece(src, WORK / "2a.mp4", t["ready"] - 0.3, t["hover"] + 0.4)
    b = piece(src, WORK / "2b.mp4", t["hover"] + 0.4, t["end"] - 0.2, crop=zoom(t["zoom_box"], 1.25))
    out.append(caption(join([(a, ""), (b, "fade")], WORK / "2.mp4"), "Models sized for your card", WORK / "2c.mp4"))

    # 3. Chat: typing at 1.5x, the answer streams at real speed, then the measured tok/s up close.
    t, src = m["chat"], REC / "chat.webm"
    a = piece(src, WORK / "3a.mp4", t["ready"] - 0.3, t["send"], speed=1.5)
    b = piece(src, WORK / "3b.mp4", t["send"], t["tps_shown"] + 0.3)
    c = piece(src, WORK / "3c.mp4", t["tps_shown"] + 0.3, t["end"] - 0.3, crop=zoom(t["zoom_box"], 2.2))
    out.append(caption(join([(a, ""), (b, "cut"), (c, "fade")], WORK / "3.mp4"), "Streams on a B60", WORK / "3c2.mp4"))

    # 4. Work: model work between the visible steps runs 4x; the question, each approval and the
    # result play at real speed.
    t, src = m["work"], REC / "work.webm"
    spans = [
        (t["ready"] - 0.3, t["send"], 2.0),            # typing the task
        (t["send"], t["question"] - 0.4, 4.0),          # the model reads the task and decides to ask
        (t["question"] - 0.4, t["answered"] + 0.3, 1.0),  # the question, and the click on Python
        (t["answered"] + 0.3, t["approval1"], 4.0),
        (t["approval1"], t["allowed1"] + 0.2, 1.0),     # "Allow Arc Llama to write a file?"
        (t["allowed1"] + 0.2, t["approval2"], 4.0),
        (t["approval2"], t["allowed2"] + 0.2, 1.0),     # "Allow Arc Llama to run a command?"
        (t["allowed2"] + 0.2, t["result"], 4.0),
        (t["result"], t["end"] - 0.2, 1.0),             # the change, ready to review or undo
    ]
    parts = [(piece(src, WORK / f"4-{i}.mp4", s, e, sp), "cut") for i, (s, e, sp) in enumerate(spans)]
    out.append(caption(join(parts, WORK / "4.mp4"), "Agents that ask before they act", WORK / "4c.mp4"))
    # Poster: the question with its choices, half a second after it appears, before the pointer moves.
    into_work = sum(duration(p) for p, _ in parts[:2]) + 0.4 + 0.5
    poster_at = sum(duration(p) for p in out[:3]) - 3 * FADE + into_work
    return out, poster_at


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2] / "docs" / "media")
    ap.add_argument("--crf", type=int, default=22)
    ap.add_argument("--webp-fps", type=int, default=15)
    ap.add_argument("--webp-q", type=int, default=70)
    args = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    clips, poster_at = shots()
    master = join([(p, "fade") for p in clips], WORK / "master.mp4")
    run("-ss", f"{poster_at:.2f}", "-i", str(master), "-frames:v", "1", str(args.out / "demo-poster.png"))
    mp4 = args.out / "demo.mp4"
    run("-i", str(master), "-c:v", "libx264", "-crf", str(args.crf), "-preset", "slow", "-tune", "stillimage",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(mp4))
    webp = args.out / "demo.webp"
    run("-i", str(master), "-vf", f"fps={args.webp_fps},scale=960:-2:flags=lanczos", "-c:v", "libwebp_anim",
        "-lossless", "0", "-q:v", str(args.webp_q), "-compression_level", "6", "-loop", "0", "-an", str(webp))
    print(json.dumps({"length_s": round(duration(master), 2),
                      "mp4_mb": round(mp4.stat().st_size / 2**20, 2), "webp_mb": round(webp.stat().st_size / 2**20, 2),
                      "poster_at_s": round(poster_at, 2)}))


if __name__ == "__main__":
    main()
