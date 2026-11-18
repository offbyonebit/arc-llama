#!/usr/bin/env python3
"""Beat-cut trailer using record.mjs takes; --placeholder needs no GPU or app."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from edit import duration, run, zoom

FPS = 30
ROOT = Path(__file__).resolve().parents[2]
REC = Path(os.environ.get("DEMO", "/tmp/arc-demo")) / "rec" / "trailer"


def stamp(seconds: float) -> str:
    cs = round(seconds * 100)
    return f"{cs // 360000}:{cs // 6000 % 60:02}:{cs // 100 % 60:02}.{cs % 100:02}"


def safe(text: str) -> str:
    return text.replace("\\", " ").replace("{", "(").replace("}", ")").replace("\n", " ")


def subtitles(path: Path, length: float, events: list) -> None:
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1440
PlayResY: 900
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Main,Inter,72,&H00FFFFFF,&H00FFFFFF,&H80FDC700,&H00000000,-1,0,0,0,100,100,0,0,1,3,0,5,30,30,30,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    for start, end, text, size, y, color, punch in events:
        tags = f"\\pos(720,{y})\\b800\\fs{size}\\c&H{color}&\\blur2"
        if punch:
            tags += "\\fscx110\\fscy110\\t(0,133,\\fscx100\\fscy100)"
        lines.append(f"Dialogue: 1,{stamp(start)},{stamp(min(end, length))},Main,,0,0,0,,{{{tags}}}{safe(text)}")
    path.write_text(header + "\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--placeholder", action="store_true")
    ap.add_argument("--with-27b", action="store_true", help="include optional bigger take (also in placeholder)")
    ap.add_argument("--music", type=Path)
    ap.add_argument("--bpm", type=float, default=120)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--rec", type=Path, default=REC)
    args = ap.parse_args()
    if not math.isfinite(args.bpm) or not 30 <= args.bpm <= 300:
        ap.error("--bpm must be between 30 and 300")
    out = (args.out or (Path("/tmp/arc-trailer-placeholder") if args.placeholder else ROOT / "docs/media")).resolve()
    if args.placeholder and (out == ROOT / "docs/media" or ROOT / "docs/media" in out.parents):
        ap.error("placeholder output must stay outside docs/media")
    out.mkdir(parents=True, exist_ok=True)
    work = out / ".trailer-work"
    work.mkdir(exist_ok=True)
    rec = args.rec
    names = ["terminal", "first-run", "speed", "thinking", "work", "montage"]
    if args.with_27b:
        names.append("bigger")
    if args.placeholder:
        rec = work / "rec" / "trailer"
        rec.mkdir(exist_ok=True)
        for name in names:
            data = dict(tps="PLACEHOLDER", version="PLACEHOLDER", gpu=dict(name="PLACEHOLDER", vram_mb="PLACEHOLDER"),
                        backend="PLACEHOLDER", ready=0, download=0, loaded=32, first_token=0,
                        browser=8, type=0, end=40, saved=1, question=1, answered=2, approval1=3,
                        discover=0, hermes=1, settings=2, models=3, speculative_active=False)
            (rec / f"{name}.json").write_text(json.dumps(data, indent=2))
            label = work / f"{name}-label.ass"
            subtitles(label, 40, [(0, 40, name.upper() + " / PLACEHOLDER", 32, 110, "FFFFFF", False)])
            run("-f", "lavfi", "-i", "color=c=0x0b101a:s=1440x900:r=30:d=40", "-vf", f"ass='{label}'",
                "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22", str(rec / f"{name}.mp4"))
    marks = {name: json.loads((rec / f"{name}.json").read_text()) for name in names}
    setup = marks["first-run"]
    for key in ("version", "backend", "gpu"):
        if not setup.get(key):
            raise ValueError(f"first-run: missing {key}")
    for name in ("speed", "bigger") if args.with_27b else ("speed",):
        m = marks[name]
        value = m.get("tps")
        if not (args.placeholder and value == "PLACEHOLDER") and (
            not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0
        ):
            raise ValueError(f"{name}: missing/invalid measured tok/s; record a new take")
        if not m.get("gpu", {}).get("name") or not m["gpu"].get("vram_mb"):
            raise ValueError(f"{name}: missing GPU name/VRAM from app")
    beat = 60 / args.bpm
    # All boundaries are frame-quantized beats, including variable download length.
    def at(beats):
        return round(beats * beat * FPS) / FPS

    events, cuts, parts = [], [], []
    cursor = 0.0

    def text(start, end, value, size=72, y=700, blue=False, punch=True):
        events.append((start, end, value, size, y, "FDC700" if blue else "FFFFFF", punch))

    def clip(name, beats, start=0, speed=1, crop=None, blur=False):
        nonlocal cursor
        end = at(cursor + beats)
        begin = at(cursor)
        length = end - begin
        dst = work / f"part-{len(parts)}.mp4"
        vf = [f"setpts=(PTS-STARTPTS)/{speed}"]
        if crop:
            vf.append(crop)
        vf += ["scale=1440:900", "setsar=1", "fps=30"]
        if blur:
            vf += ["gblur=sigma=12", "eq=brightness=-0.16"]
        vf += [f"tpad=stop_mode=clone:stop_duration={length}", "format=yuv420p"]
        if name is None:
            inputs = ["-f", "lavfi", "-i", f"color=c=0x05080e:s=1440x900:r=30:d={length}"]
        else:
            source = rec / f"{name}.mp4"
            # Padding may hold a finished frame, but must never replace missing footage.
            if start < 0 or duration(source) <= start:
                raise ValueError(f"{name}: footage does not cover {start}")
            inputs = ["-ss", str(start), "-t", str(length * speed), "-i", str(source)]
        run(*inputs, "-an", "-vf", ",".join(vf), "-t", str(length), "-c:v", "libx264",
            "-crf", "16", "-preset", "veryfast", str(dst))
        parts.append(dst)
        cuts.append(begin)
        cursor += beats
        return begin, end

    def counter(name, beats):
        m = marks[name]
        begin, end = clip(name, beats, m["first_token"], blur=True)
        for frame in range(round((end - begin) * FPS)):
            elapsed = frame / FPS
            value = "##.#" if args.placeholder else f"{m['tps'] * min(1, elapsed / .8):.1f}"
            text(begin + elapsed, begin + elapsed + 1 / FPS, value, 190, 390, punch=False)
        text(begin, end, "TOKENS / SEC", 38, 530)
        text(begin, end, "measured live on " + m["gpu"]["name"], 26, 600, punch=False)
        if name == "bigger":
            text(begin, begin + beat, "27B.", 64, 200)
            if m.get("speculative_active") is True and m.get("speculative_evidence"):
                text(begin + beat, end, "SPECULATIVE DECODING.", 58, 200)

    b, e = clip(None, 6)
    for i, word in enumerate(("NO CLOUD.", "NO API BILL.", "JUST YOUR ARC.")):
        text(at(i), at(i + 1) if i < 2 else e, word, 94, 450)
    t = marks["terminal"]
    b, e = clip("terminal", 4, t["type"], speed=2)
    text(b, e, "ONE COMMAND.")
    crop = zoom(t["zoom_box"], 3, t.get("scale", 1)) if "zoom_box" in t else None
    b, e = clip("terminal", 2, max(0, t["browser"] - 1), crop=crop)
    text(b, e, "ONE COMMAND.")
    # Keep the click at real speed, download only at 8x; split a long download across
    # two beat-aligned excerpts so no streaming or setup waits need a dishonest speed.
    t = setup
    b, e = clip("first-run", 2, max(0, t["download"] - .5))
    text(b, e, "ONE CLICK TO AN ANSWER.", 64)
    b, e = clip("first-run", 2, t["download"], speed=8)
    text(b, e, "ONE CLICK TO AN ANSWER.", 64)
    wipe_start = b
    b, e = clip("first-run", 4, max(t["download"], t["loaded"] - 16), speed=8)
    text(b, e, "ONE CLICK TO AN ANSWER.", 64)
    wipe = (wipe_start, e)
    counter("speed", 10 if args.with_27b else 18)
    if args.with_27b:
        counter("bigger", 8)
    b, e = clip("thinking", 3, marks["thinking"]["ready"])
    text(b, e, "YOU DECIDE HOW HARD IT THINKS.", 58, 780)
    b, e = clip("thinking", 3, marks["thinking"]["saved"])
    text(b, e, "YOU DECIDE HOW HARD IT THINKS.", 58, 780)
    t = marks["work"]
    for key, beats in (("ready", 2), ("question", 3), ("answered", 2), ("approval1", 3)):
        b, e = clip("work", beats, t[key])
        text(b, e, "AGENTS THAT ASK FIRST.", 68, 780)
    for key in ("discover", "hermes", "settings", "models"):
        b, e = clip("montage", 1, marks["montage"][key])
        text(b, e, setup["version"], 180, 450, blue=True)
    b, e = clip("montage", 2, marks["montage"]["models"])
    text(b, e, setup["version"], 180, 450, blue=True)
    b, e = clip(None, 6)
    text(b, e, "ARC LLAMA", 112, 280)
    text(b, e, setup["version"], 90, 410, blue=True)
    text(b, e, "pip install -U arc-llama", 38, 565, punch=False)
    text(b, e, "github.com/offbyonebit/arc-llama", 30, 700, punch=False)
    poster_at = b + beat
    ass = work / "type.ass"
    subtitles(ass, e, events)
    with ass.open("a") as handle:
        handle.write(
            f"Dialogue: 0,{stamp(b)},{stamp(e)},Main,,0,0,0,,"
            r"{\an7\pos(390,520)\p1\bord0\shad0\1c&H5C3000&}"
            "m 45 0 l 615 0 b 640 0 660 20 660 45 b 660 70 640 90 615 90 "
            "l 45 90 b 20 90 0 70 0 45 b 0 20 20 0 45 0\n"
        )
    listing = work / "concat.txt"
    listing.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts))
    master = out / "trailer-1.2.mp4"
    filters = (f"ass='{ass}',drawbox=x=0:y=896:w=iw:h=4:c=0x0071C5:t=fill,"
               f"drawbox=x=0:y=0:w=iw:h=ih:c=white:t=fill:enable='eq(n,{round(at(2)*FPS)})',"
               f"drawtext=font='Inter':text='━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━':fontsize=40:"
               f"fontcolor=0x00C7FD:x='-1440+1440*min(1,max(0,(t-{wipe[0]})/{wipe[1]-wipe[0]}))':y=850:"
               f"enable='between(t,{wipe[0]},{wipe[1]})',drawbox=x=0:y=896:w=iw:h=4:c=0x0071C5:t=fill")
    music = ["-stream_loop", "-1", "-i", str(args.music.resolve())] if args.music else []
    audio = ["-map", "1:a:0", "-af", f"afade=t=out:st={e - beat}:d={beat}", "-c:a", "aac"] if music else ["-an"]
    run("-f", "concat", "-safe", "0", "-i", str(listing), *music, "-map", "0:v:0", "-vf", filters,
        "-t", str(e), "-map_metadata", "-1", "-c:v", "libx264", "-crf", "16", "-preset", "slow",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", *audio, str(master))
    run("-ss", str(poster_at), "-i", str(master), "-frames:v", "1", str(out / "trailer-1.2-poster.png"))
    webp = out / "trailer-1.2.webp"
    for fps, quality, width in ((10, 60, 960), (8, 45, 800), (6, 35, 640), (4, 25, 640)):
        run("-i", str(master), "-vf", f"fps={fps},scale={width}:-2", "-an", "-c:v", "libwebp_anim",
            "-q:v", str(quality), "-loop", "0", "-map_metadata", "-1", str(webp))
        if webp.stat().st_size < 3 * 1024**2:
            break
    if webp.stat().st_size >= 3 * 1024**2:
        raise ValueError("WebP exceeds 3 MiB; lower fps/quality before publishing")
    actual = duration(master)
    assert abs(actual - e) <= 1 / FPS, (actual, e)
    assert all(abs(t / beat - round(t / beat)) * beat <= 1 / FPS for t in cuts)
    report = dict(length_s=actual, cut_times_s=cuts, bpm=args.bpm, placeholder=args.placeholder,
                  webp_bytes=webp.stat().st_size)
    (out / "timing.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.placeholder:
        for name, seconds in (("cold-open", .75 * beat), ("counter", 20 * beat + .4),
                              ("thinking", 39 * beat), ("work", 47 * beat),
                              ("montage", 55 * beat), ("end-card", poster_at)):
            run("-ss", str(seconds), "-i", str(master), "-frames:v", "1", str(out / f"{name}.png"))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
