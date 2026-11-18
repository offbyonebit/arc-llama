#!/usr/bin/env python3
"""1.2 trailer: a beat-cut launch spot from record.mjs takes; --placeholder needs no GPU or app.

Every number on screen comes from the take's JSON (tok/s, GPU, VRAM, version, backend, draft stats).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import tempfile
from pathlib import Path

from edit import duration, run

FPS, W, H = 30, 1440, 900
ROOT = Path(__file__).resolve().parents[2]
REC = Path(os.environ.get("DEMO", "/tmp/arc-demo")) / "rec" / "trailer"
WHITE, BLUE, DEEP = "FFFFFF", "FDC700", "C57100"  # ASS is BGR: #00C7FD, #0071C5
BG = "0x04070d"


def stamp(seconds: float) -> str:
    cs = max(0, round(seconds * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02}:{cs // 100 % 60:02}.{cs % 100:02}"


def safe(text: str) -> str:
    return text.replace("\\", " ").replace("{", "(").replace("}", ")").replace("\n", " ")


def esc(path: Path) -> str:
    return path.as_posix().replace(":", "\\:")


def gpu_label(gpu: dict) -> str:
    name = re.sub(r"\((R|TM)\)|\bIntel\b|\bGraphics\b", "", str(gpu["name"]), flags=re.I)
    return " ".join(name.split()).upper()


def draft_rate(lines: list) -> int | None:
    """Accepted / generated draft tokens from llama.cpp's MTP statistics line, if the take logged one."""
    for line in reversed(lines or []):
        acc, gen = re.search(r"#acc tokens\s*=\s*(\d+)", line), re.search(r"#gen tokens\s*=\s*(\d+)", line)
        if acc and gen and int(gen[1]) > 0:
            return min(100, round(100 * int(acc[1]) / int(gen[1])))
    return None


class Type:
    """ASS events: slams, labels, counters, streaks and rings."""

    def __init__(self):
        self.lines: list[str] = []

    def add(self, start, end, tags, body, layer=5):
        self.lines.append(f"Dialogue: {layer},{stamp(start)},{stamp(end)},Main,,0,0,0,,{{{tags}}}{body}")

    def slam(self, start, end, text, size, x, y, an=5, color=WHITE, glow=True):
        pos = f"\\an{an}\\pos({x},{y})\\fs{size}\\fsp-2\\b900"
        text = safe(text)
        # Cyan ghost: an offset copy that snaps in large and fades, like an RGB split.
        self.add(start, min(end, start + .3), pos.replace(f"\\pos({x},{y})", f"\\pos({x + 12},{y})") +
                 f"\\c&H{BLUE}&\\alpha&H30&\\fscx165\\fscy165\\t(0,140,\\fscx100\\fscy100)\\t(80,300,\\alpha&HFF&)", text, 3)
        if glow:
            self.add(start, end, pos + f"\\1a&HFF&\\3c&H{DEEP}&\\bord14\\blur16\\fscx140\\fscy140\\t(0,120,\\fscx100\\fscy100)", text, 4)
        self.add(start, end, pos + f"\\c&H{color}&\\bord0\\blur.6\\alpha&HFF&\\fscx140\\fscy140\\fsp40"
                 f"\\t(0,50,\\alpha&H00&)\\t(0,120,\\fscx100\\fscy100\\fsp-2)", text)

    def label(self, start, end, text, size, x, y, an=5, color=WHITE, alpha="00", spacing=6):
        self.add(start, end, f"\\an{an}\\pos({x},{y})\\fs{size}\\b700\\fsp{spacing}\\c&H{color}&\\alpha&H{alpha}&\\bord0\\fad(90,0)", safe(text))

    def rect(self, start, end, x, y, w, h, color, alpha="00", grow=0, blur=0, layer=2):
        grow_tags = f"\\fscx0\\t(0,{grow},\\fscx100)" if grow else ""
        self.add(start, end, f"\\an7\\pos({x},{y})\\p1\\bord0\\shad0\\blur{blur}\\c&H{color}&\\alpha&H{alpha}&{grow_tags}",
                 f"m 0 0 l {w} 0 l {w} {h} l 0 {h}", layer)

    def kicker(self, start, end, number, word):
        self.rect(start, end, 96, 726, 64, 6, BLUE, grow=160)
        self.label(start, end, f"{number:02} / {word}", 24, 176, 729, an=4, color=BLUE)

    def headline(self, start, end, text, size=96):
        self.slam(start, end, text, size, 92, 838, an=1)

    def ring(self, t, x, y, size=260):
        circle = "m 0 -100 b 55 -100 100 -55 100 0 b 100 55 55 100 0 100 b -55 100 -100 55 -100 0 b -100 -55 -55 -100 0 -100"
        self.add(t, t + .45, f"\\an7\\pos({x},{y})\\p1\\1a&HFF&\\3c&H{BLUE}&\\bord3\\blur1\\fscx{size // 4}\\fscy{size // 4}"
                 f"\\t(0,450,\\fscx{size * 2}\\fscy{size * 2}\\alpha&HFF&)", circle, 6)

    def streaks(self, start, end, y0, y1, seed, count=40):
        rnd = random.Random(seed)
        for _ in range(count):
            t = start + rnd.random() * max(0.01, end - start - .3)
            y, length, life = rnd.randint(y0, y1), rnd.randint(140, 520), rnd.uniform(.18, .38)
            alpha = rnd.choice(("30", "60", "90"))
            self.add(t, t + life, f"\\an7\\move(-600,{y},{W + 80},{y})\\p1\\bord0\\blur1.5\\c&H{BLUE}&\\alpha&H{alpha}&",
                     f"m 0 0 l {length} 0 l {length} 3 l 0 3", 1)

    def write(self, path: Path) -> None:
        header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 2
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Main,Inter,72,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,0,0,5,0,0,0,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        path.write_text(header + "\n".join(self.lines) + "\n", encoding="utf-8")


def placeholder_takes(rec: Path, work: Path, names: list) -> None:
    for name in names:
        data = dict(tps="PLACEHOLDER", version="PLACEHOLDER", gpu=dict(name="PLACEHOLDER", vram_mb="PLACEHOLDER"),
                    backend="PLACEHOLDER", ready=0, download=2, loaded=30, first_token=0, type=0, browser=8,
                    saved=1, question=1, answered=2, approval1=3, discover=0, hermes=1, settings=2, models=3,
                    speculative_active=False)
        (rec / f"{name}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        label = work / f"{name}-label.ass"
        t = Type()
        t.label(0, 40, name.upper() + " / PLACEHOLDER", 32, 720, 110)
        t.write(label)
        run("-f", "lavfi", "-i", f"color=c=0x0b101a:s={W}x{H}:r=30:d=40", "-vf", f"ass='{esc(label)}'",
            "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22", str(rec / f"{name}.mp4"))


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
    out = (args.out or ((Path(tempfile.gettempdir()) / "arc-trailer-placeholder") if args.placeholder else ROOT / "docs/media")).resolve()
    if args.placeholder and (out == ROOT / "docs/media" or ROOT / "docs/media" in out.parents):
        ap.error("placeholder output must stay outside docs/media")
    work = out / ".trailer-work"
    work.mkdir(parents=True, exist_ok=True)
    big = args.with_27b
    names = ["terminal", "first-run", "speed", "thinking", "work", "montage"] + (["bigger"] if big else [])
    rec = args.rec
    if args.placeholder:
        rec = work / "rec" / "trailer"
        rec.mkdir(parents=True, exist_ok=True)
        placeholder_takes(rec, work, names)
    marks = {name: json.loads((rec / f"{name}.json").read_text(encoding="utf-8")) for name in names}
    setup = marks["first-run"]
    for key in ("version", "backend", "gpu"):
        if not setup.get(key):
            raise ValueError(f"first-run: missing {key}")
    fake = args.placeholder
    for name in ("speed", "bigger") if big else ("speed",):
        m = marks[name]
        value = m.get("tps")
        if not (fake and value == "PLACEHOLDER") and (
            not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0
        ):
            raise ValueError(f"{name}: missing/invalid measured tok/s; record a new take")
        if not m.get("gpu", {}).get("name") or not m["gpu"].get("vram_mb"):
            raise ValueError(f"{name}: missing GPU name/VRAM from app")
    gpu = marks["speed"]["gpu"]
    card = "ARC PRO B60 (PLACEHOLDER)" if fake else gpu_label(gpu)
    vram = "## GB" if fake else f"{min((4, 6, 8, 10, 11, 12, 16, 20, 24, 32, 48, 64, 96, 128), key=lambda gb: abs(gb - gpu['vram_mb'] / 1024))} GB"
    version = "#.#" if fake else str(setup["version"])
    short = version if fake else ".".join(version.split(".")[:2])
    backend = "VULKAN/SYCL" if fake else str(setup["backend"]).upper()

    beat = 60 / args.bpm

    def at(beats):
        return round(beats * beat * FPS) / FPS

    ty = Type()
    cuts, parts, flashes, shakes = [], [], [], []
    cursor = 0

    def clip(name, beats, start=0.0, speed=1.0, zoom=(1.0, 1.0), focus=(720, 450), trails=False, dim=0.0, glitch=True,
             blur=0, panel=0):
        """One beat-aligned piece; ``focus`` is a page point (1440x900 basis) the push-in centres on.

        ``blur`` softens the whole frame and ``panel`` frosts a dark strip down the left, so app
        text (such as the app's own tok/s readout) can never be read behind a measured number.
        """
        nonlocal cursor
        begin, end = at(cursor), at(cursor + beats)
        length = end - begin
        dst = work / f"part-{len(parts):02}.mp4"
        z0, z1 = zoom
        z = f"({z0}+({z1 - z0})*(1-pow(1-min(t/{length},1),3)))"
        fx, fy = focus
        vf = [f"setpts=(PTS-STARTPTS)/{speed}", "fps=30",
              f"scale=w='2*trunc({W}*{z}/2)':h='2*trunc({H}*{z}/2)':eval=frame:flags=bicubic",
              f"crop={W}:{H}:x='max(0,min({W}*{z}-{W},{fx}*{z}-{W // 2}))':y='max(0,min({H}*{z}-{H},{fy}*{z}-{H // 2}))'",
              "setsar=1"]
        if trails:
            vf.append("lagfun=decay=0.84")
        if dim:
            vf.append(f"eq=brightness={-dim}:saturation=1.15")
        if blur:
            vf.append(f"gblur=sigma={blur}")
        if panel:
            vf.append(f"split[a][b];[b]crop={panel}:{H}:0:0,gblur=sigma=30,eq=brightness=-0.4[p];[a][p]overlay=0:0")
        if glitch:
            vf += ["rgbashift=rh=-18:bh=18:enable='lt(t,0.07)'", "noise=alls=45:allf=t:enable='lt(t,0.07)'"]
        vf += [f"tpad=stop_mode=clone:stop_duration={length}", "format=yuv420p"]
        if name is None:
            inputs = ["-f", "lavfi", "-i", f"color=c={BG}:s={W}x{H}:r=30:d={length}"]
        else:
            source = rec / f"{name}.mp4"
            # Padding may hold a finished frame, but must never replace missing footage.
            if start < 0 or duration(source) <= start:
                raise ValueError(f"{name}: footage does not cover {start}")
            inputs = ["-ss", str(start), "-t", str(length * speed), "-i", str(source)]
        run(*inputs, "-an", "-vf", ",".join(vf), "-t", str(length), "-c:v", "libx264",
            "-crf", "14", "-preset", "veryfast", str(dst))
        parts.append(dst)
        cuts.append(begin)
        cursor += beats
        return begin, end

    def box_focus(box, fallback):
        return (box["x"] + box["w"] / 2, box["y"] + box["h"] / 2) if isinstance(box, dict) and "x" in box else fallback

    def hud(begin, end):
        ty.label(begin, end, f"● LIVE CAPTURE  ·  {card}", 18, 40, 40, an=7, color=BLUE, alpha="20", spacing=4)
        ty.label(begin, end, f"ARC LLAMA {version}", 18, W - 40, 40, an=9, alpha="50", spacing=4)

    def counter(name, beats, number, word):
        m = marks[name]
        focus = box_focus(m.get("stream_box"), (900, 420))
        begin, end = clip(name, beats, m["first_token"], zoom=(1.35, 1.7), focus=focus, trails=True, dim=.12,
                          blur=11, panel=840)
        hud(begin, end)
        ty.rect(begin, end, 838, 0, 3, H, DEEP, alpha="30", blur=0, layer=0)
        ty.kicker(begin, end, number, word)
        ramp, x, y = 1.0, 96, 440
        hit = begin + ramp
        ty.streaks(begin, hit + .6, 250, 620, seed=number)
        if fake:
            ty.slam(begin, end, "##.#", 250, x, y, an=4)
        else:
            for frame in range(round(ramp * FPS)):
                p = frame / (ramp * FPS)
                value = m["tps"] * (1 - (1 - p) ** 4)
                ty.add(begin + frame / FPS, begin + (frame + 1) / FPS,
                      f"\\an4\\pos({x},{y})\\fs250\\b900\\fsp-6\\c&H{WHITE}&\\bord0\\blur.6", f"{value:.1f}")
            ty.add(hit, end, f"\\an4\\pos({x},{y})\\fs250\\b900\\fsp-6\\c&H{BLUE}&\\bord0\\blur.6\\fscx118\\fscy118"
                  f"\\t(0,160,\\fscx100\\fscy100\\c&H{WHITE}&)", f"{m['tps']:.1f}")
            ty.add(hit, end, f"\\an4\\pos({x},{y})\\fs250\\b900\\fsp-6\\1a&HFF&\\3c&H{DEEP}&\\bord16\\blur18", f"{m['tps']:.1f}", 4)
        ty.ring(hit, 300, y)
        flashes.append(hit)
        shakes.append(hit)
        ty.label(hit, end, "TOKENS / SEC", 34, x + 6, y + 150, an=4, color=BLUE, spacing=10)
        return begin, end, hit

    # 1. Cold open: three slams on black, the last one in Intel blue.
    clip(None, 4, glitch=False)
    for i, (word, color) in enumerate((("NO CLOUD.", WHITE), ("NO API BILL.", WHITE), ("JUST YOUR ARC.", BLUE))):
        ty.slam(at(i), at(i + 1 if i < 2 else 4), word, 150, W // 2, H // 2, color=color)
        shakes.append(at(i))
    flashes.append(at(2))
    # 2. The card, as the app detected it.
    b, e = clip(None, 2)
    ty.label(b, e, "DETECTED LIVE", 24, W // 2, 270, color=BLUE, spacing=14)
    ty.slam(b, e, card, 128, W // 2, 420)
    ty.slam(b + beat, e, vram, 170, W // 2, 600, color=BLUE)
    flashes.append(b)
    shakes.append(b)
    # 3. One command: the terminal types, then a hard cut onto the detected-device line.
    t = marks["terminal"]
    b, e = clip("terminal", 4, t.get("type", 0), speed=2, zoom=(1.15, 1.45), focus=(330, 160))
    hud(b, e)
    ty.kicker(b, e, 1, "START")
    ty.headline(b, e, "ONE COMMAND.")
    b2, e2 = clip("terminal", 2, max(0, t["browser"] - .5), zoom=(2.2, 2.45), focus=box_focus(t.get("zoom_box"), (330, 250)))
    hud(b2, e2)
    ty.kicker(b2, e2, 1, "START")
    ty.headline(b2, e2, "ONE COMMAND.")
    # 4. First run: the click at real speed, the download as a labelled time-lapse, then the model loads.
    t = setup
    download_at = t.get("download", 0)
    button = box_focus(t.get("button_box"), (380, 420))
    b, e = clip("first-run", 2, max(0, download_at - .7), zoom=(1.3, 1.6), focus=button)
    hud(b, e)
    ty.kicker(b, e, 2, "FIRST RUN")
    ty.headline(b, e, "ONE CLICK.")
    lapse_beats = 4 if big else 6
    span = max(t["loaded"] - download_at, 0.1)
    speed = max(1.0, span / (lapse_beats * beat))
    b, e = clip("first-run", lapse_beats, download_at, speed=speed, zoom=(1.0, 1.2), focus=(720, 400))
    hud(b, e)
    ty.kicker(b, e, 2, "FIRST RUN")
    ty.headline(b, e, "ONE CLICK.")
    if speed >= 2:
        ty.label(b, e, f"▸▸ DOWNLOAD ×{speed:.0f}", 22, W - 40, 80, an=9, color=BLUE, spacing=4)
    b, e = clip("first-run", 2, t["loaded"], zoom=(1.2, 1.35), focus=(720, 450))
    hud(b, e)
    ty.kicker(b, e, 2, "FIRST RUN")
    ty.headline(b, e, "TO AN ANSWER.")
    # 5. Speed: the answer streams at real speed while the measured tok/s counts up and lands.
    speed_at, e, _ = counter("speed", 10 if big else 12, 3, "SPEED")
    ty.label(speed_at + 1.2, e, f"MEASURED LIVE  ·  {card}  ·  {backend}", 22, 102, 650, an=4, alpha="20", spacing=5)
    if big:
        m = marks["bigger"]
        b, e, hit = counter("bigger", 8, 4, "GO BIGGER")
        ty.add(b, e, f"\\an6\\pos({W - 60},{H // 2})\\fs360\\b900\\fsp-14\\1a&HFF&\\3c&H{BLUE}&\\bord3\\3a&H50&"
              f"\\fscx112\\fscy112\\t(0,{round((e - b) * 1000)},\\fscx100\\fscy100)", "27B", 1)
        ty.slam(b, e, "27B.", 72, 102, 300, an=4)
        rate = draft_rate(m.get("speculative_evidence")) if m.get("speculative_active") is True else None
        if m.get("speculative_active") is True and m.get("speculative_evidence"):
            line = "SPECULATIVE DECODING" + (f"  ·  {rate}% OF DRAFT TOKENS ACCEPTED" if rate is not None else "")
            ty.label(hit + .2, e, line, 22, 102, 650, an=4, color=BLUE, spacing=5)
    # 6. Thinking budget: push into the settings panel as 256 goes in and saves.
    th = marks["thinking"]
    field = box_focus(th.get("budget_box"), (1180, 420))
    thinking_at, e = clip("thinking", 3, th["ready"], zoom=(1.25, 1.6), focus=field)
    hud(thinking_at, e)
    ty.kicker(thinking_at, e, 5, "CONTROL")
    ty.headline(thinking_at, e, "YOU SET HOW", 88)
    b, e = clip("thinking", 3, th["saved"], zoom=(1.6, 1.75), focus=field)
    hud(b, e)
    ty.kicker(b, e, 5, "CONTROL")
    ty.headline(b, e, "HARD IT THINKS.", 88)
    # 7. Work: one word per hard cut.
    wk = marks["work"]
    approval = box_focus(wk.get("approval_box"), (720, 470))
    work_at = at(cursor)
    shots = (("ready", "AGENTS", (1.2, 1.35), (720, 700)), ("question", "THAT ASK", (1.35, 1.6), (720, 430)),
             ("answered", "BEFORE", (1.3, 1.45), (720, 470)), ("approval1", "THEY ACT.", (1.5, 1.8), approval))
    for (key, word, zoom, focus), beats in zip(shots, (2, 2, 2, 2) if big else (2, 3, 2, 3), strict=True):
        b, e = clip("work", beats, wk[key], zoom=zoom, focus=focus)
        hud(b, e)
        ty.kicker(b, e, 6, "WORK")
        ty.headline(b, e, word)
    # 8. Montage: a cut a beat, "1.2" punching on each.
    mo = marks["montage"]
    montage_at = at(cursor)
    flips = [("montage", mo["discover"], (1.1, 1.25)), ("montage", mo["hermes"], (1.3, 1.45)),
             ("montage", mo["settings"], (1.2, 1.35)), ("montage", mo["models"], (1.1, 1.25)),
             ("speed", marks["speed"]["first_token"] + 1.5, (2.0, 2.2)), ("terminal", max(0, marks["terminal"]["browser"] - .5), (2.2, 2.4))]
    if not big:
        flips += [("work", wk["question"], (1.6, 1.8)), ("first-run", setup["loaded"], (1.4, 1.55))]
    for name, start, zoom in flips:
        b, e = clip(name, 1, start, zoom=zoom, dim=.3, blur=13)
        ty.slam(b, e, short, 300, W // 2, H // 2, color=BLUE)
        shakes.append(b)
    # 9. End card.
    b, e = clip(None, 8)
    ty.slam(b, e, "ARC LLAMA", 176, W // 2, 330)
    ty.slam(b + beat, e, short, 132, W // 2, 500, color=BLUE)
    ty.rect(b + 2 * beat, e, 420, 590, 600, 76, "2A1A0C", blur=0, grow=180)
    ty.label(b + 2 * beat, e, "pip install -U arc-llama", 34, W // 2, 628, spacing=1)
    ty.label(b + 3 * beat, e, "github.com/offbyonebit/arc-llama", 26, W // 2, 740, alpha="30", spacing=3)
    flashes.append(b)
    shakes.append(b)
    total = e
    poster_at = b + 4 * beat
    ass = work / "type.ass"
    ty.write(ass)

    listing = work / "concat.txt"
    listing.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
    master = out / "trailer-1.2.mp4"
    flash = "+".join(f"eq(n,{round(t * FPS)})" for t in flashes)
    after = "+".join(f"eq(n,{round(t * FPS) + 1})" for t in flashes)
    jolt = "+".join(f"between(t,{t:.3f},{t + .35:.3f})*sin(95*(t-{t:.3f}))*exp(-9*(t-{t:.3f}))" for t in shakes)
    filters = ",".join([
        f"ass='{esc(ass)}'",
        f"crop={W - 32}:{H - 20}:x='16+14*({jolt})':y='10+8*({jolt})',scale={W}:{H}",
        "eq=contrast=1.12:saturation=1.12", "colorbalance=bs=.07:bm=.03",
        "vignette=angle=PI/4.6", "drawgrid=w=iw:h=4:t=1:c=black@0.2",
        f"drawbox=x=0:y={H - 4}:w=iw:h=4:c=0x0071C5:t=fill",
        f"drawbox=x=0:y=0:w=iw:h=ih:c=white@0.85:t=fill:enable='{flash}'",
        f"drawbox=x=0:y=0:w=iw:h=ih:c=white@0.3:t=fill:enable='{after}'",
    ])
    music = ["-stream_loop", "-1", "-i", str(args.music.resolve())] if args.music else []
    audio = ["-map", "1:a:0", "-af", f"afade=t=out:st={total - beat}:d={beat}", "-c:a", "aac"] if music else ["-an"]
    run("-f", "concat", "-safe", "0", "-i", str(listing), *music, "-map", "0:v:0", "-vf", filters,
        "-t", str(total), "-map_metadata", "-1", "-c:v", "libx264", "-crf", "18", "-preset", "slow",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", *audio, str(master))
    run("-ss", str(poster_at), "-i", str(master), "-frames:v", "1", str(out / "trailer-1.2-poster.png"))
    webp = out / "trailer-1.2.webp"
    for fps, quality, width in ((15, 60, 960), (12, 50, 960), (10, 45, 800), (8, 35, 640)):
        run("-i", str(master), "-vf", f"fps={fps},scale={width}:-2", "-an", "-c:v", "libwebp_anim",
            "-q:v", str(quality), "-loop", "0", "-map_metadata", "-1", str(webp))
        if webp.stat().st_size < 3 * 1024**2:
            break
    if webp.stat().st_size >= 3 * 1024**2:
        raise ValueError("WebP exceeds 3 MiB; lower fps/quality before publishing")
    actual = duration(master)
    assert abs(actual - total) <= 1 / FPS, (actual, total)
    assert all(abs(t / beat - round(t / beat)) * beat <= 1 / FPS for t in cuts)
    report = dict(length_s=actual, cut_times_s=cuts, bpm=args.bpm, placeholder=fake, webp_bytes=webp.stat().st_size)
    (out / "timing.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    grabs = work / "grabs"
    grabs.mkdir(exist_ok=True)
    for name, seconds in (("cold-open", at(2) + .2), ("card", at(4) + beat + .3), ("terminal", at(6) + .9),
                          ("counter", speed_at + 1.6), ("thinking", thinking_at + 1.0), ("work", work_at + 3.4),
                          ("montage", montage_at + .3), ("end-card", poster_at)):
        run("-ss", str(seconds), "-i", str(master), "-frames:v", "1", str(grabs / f"{name}.png"))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
