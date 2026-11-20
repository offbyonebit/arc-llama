#!/usr/bin/env python3
"""1.2 trailer: a "what got better" spot cut from record.mjs takes; --placeholder needs no GPU or app.

Live numbers (tok/s, GPU, version, draft stats) come from the take's JSON. The before/after numbers
are the 1.2.0 release measurements on an Arc Pro B60, kept in WINS below.
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
DIM = "7A6E66"
# Measured on an Arc Pro B60 for the 1.2.0 release notes: service start with six GGUFs and first
# model load after a restart (seconds, 1.1.1 then 1.2), and MTP decode against plain decode on SYCL.
WINS = dict(start=(66.8, 6.1), load=(21.0, 10.8), mtp=1.9)


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

    def hold(self, start, end, text, size, x, y, an=5, color=WHITE):
        """A slam's settled frame, for a caption that carries on across a hard cut."""
        pos = f"\\an{an}\\pos({x},{y})\\fs{size}\\fsp-2\\b900"
        self.add(start, end, pos + f"\\1a&HFF&\\3c&H{DEEP}&\\bord14\\blur16", safe(text), 4)
        self.add(start, end, pos + f"\\c&H{color}&\\bord0\\blur.6", safe(text))

    def label(self, start, end, text, size, x, y, an=5, color=WHITE, alpha="00", spacing=6):
        self.add(start, end, f"\\an{an}\\pos({x},{y})\\fs{size}\\b700\\fsp{spacing}\\c&H{color}&\\alpha&H{alpha}&\\bord0\\fad(90,0)", safe(text))

    def rect(self, start, end, x, y, w, h, color, alpha="00", grow=0, blur=0, layer=2):
        grow_tags = f"\\fscx0\\t(0,{grow},\\fscx100)" if grow else ""
        self.add(start, end, f"\\an7\\pos({x},{y})\\p1\\bord0\\shad0\\blur{blur}\\c&H{color}&\\alpha&H{alpha}&{grow_tags}",
                 f"m 0 0 l {w} 0 l {w} {h} l 0 {h}", layer)

    def kicker(self, start, end, number, word, y=726):
        self.rect(start, end, 96, y, 64, 6, BLUE, grow=160)
        self.label(start, end, f"{number:02} / {word}", 24, 176, y + 3, an=4, color=BLUE)

    def headline(self, start, end, text, size=96, y=838, settle=False):
        (self.hold if settle else self.slam)(start, end, text, size, 92, y, an=1)

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
    names = ["terminal", "first-run", "thinking", "bigger"]
    rec = args.rec
    if args.placeholder:
        rec = work / "rec" / "trailer"
        rec.mkdir(parents=True, exist_ok=True)
        placeholder_takes(rec, work, names)
    marks = {name: json.loads((rec / f"{name}.json").read_text(encoding="utf-8")) for name in names}
    setup, m = marks["first-run"], marks["bigger"]
    if not setup.get("version"):
        raise ValueError("first-run: missing version")
    fake = args.placeholder
    value = m.get("tps")
    if not (fake and value == "PLACEHOLDER") and (
        not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0
    ):
        raise ValueError("bigger: missing/invalid measured tok/s; record a new take")
    if not m.get("gpu", {}).get("name"):
        raise ValueError("bigger: missing GPU name from app")
    card = "ARC PRO B60 (PLACEHOLDER)" if fake else gpu_label(m["gpu"])
    version = "#.#" if fake else str(setup["version"])
    short = version if fake else ".".join(version.split(".")[:2])

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

    def lines(begin, end, first, second, size=80, settle=False):
        """Two caption rows at the bottom left, the second in Intel blue."""
        (ty.hold if settle else ty.slam)(begin, end, first, size, 92, 752, an=1)
        (ty.hold if settle else ty.slam)(begin, end, second, size, 92, 838, an=1, color=BLUE)

    def race(begin, end, before, after, race_s, note):
        """1.1.1 and 1.2 bars growing at the same rate, so the 1.2 bar stops early; each value lands as its bar stops."""
        x0, width, scale = 250, 1000, 1000 / before
        for i, (label, value, color) in enumerate((("BEFORE", before, DIM), (short, after, BLUE))):
            y, length = 470 + i * 130, max(8, round(value * scale))
            stop = begin + beat + race_s * value / before
            ty.label(begin, end, label, 30, 96, y, an=4, color=color, spacing=4)
            ty.rect(begin, end, x0, y - 30, width, 60, "1A120C", layer=1)
            ty.rect(begin + beat, end, x0, y - 30, length, 60, color, grow=round(race_s * value / before * 1000))
            ty.slam(stop, end, f"{value:.1f} s", 52, x0 + length + 24, y, an=4, color=color, glow=color == BLUE)
            shakes.append(stop)
        flashes.append(begin + beat + race_s * after / before)
        ty.label(begin + beat, end, f"{note}  ·  ARC PRO B60  ·  MEASURED", 20, W - 60, 772, an=6, alpha="40", spacing=4)

    # 1. Open: the version, then what this spot is about.
    b, e = clip(None, 6, glitch=False)
    ty.label(b, e, "ARC LLAMA", 34, W // 2, 250, spacing=18, color="B4AAA0")
    ty.slam(b, e, short, 300, W // 2, 430, color=BLUE)
    ty.slam(b + 2 * beat, e, "WHAT GOT BETTER.", 84, W // 2, 640)
    flashes.append(b + 2 * beat)
    shakes += [b, b + 2 * beat]
    # 2. Startup: the 1.1.1 bar runs on long after 1.2 is ready.
    before, after = WINS["start"]
    b, e = clip(None, 10)
    ty.kicker(b, e, 1, "STARTUP", y=120)
    ty.slam(b, e, f"READY IN {after:.0f} SECONDS.", 96, 92, 250, an=1)
    race(b, e, before, after, 2.75, "SERVICE START, SIX MODELS ON DISK")
    ty.slam(b + 7 * beat, e, f"{before / after:.0f}× FASTER", 64, 92, 790, an=1, color=BLUE)
    # 3. First model load after a restart.
    before, after = WINS["load"]
    b, e = clip(None, 8)
    ty.kicker(b, e, 2, "FIRST LOAD", y=120)
    ty.slam(b, e, "HALF THE WAIT." if before / after >= 1.9 else f"{before / after:.1f}× FASTER.", 96, 92, 250, an=1)
    race(b, e, before, after, 2.5, "FIRST MODEL LOAD AFTER A RESTART")
    # 4. MTP: the 27B answer streams while its measured tok/s counts up and lands.
    focus = box_focus(m.get("stream_box"), (900, 420))
    b, e = clip("bigger", 8, m["first_token"], zoom=(1.35, 1.7), focus=focus, trails=True, dim=.12, blur=11, panel=840)
    hud(b, e)
    mtp_at = b
    ty.rect(b, e, 838, 0, 3, H, DEEP, alpha="30", layer=0)
    ty.add(b, e, f"\\an6\\pos({W - 60},{H // 2})\\fs360\\b900\\fsp-14\\1a&HFF&\\3c&H{BLUE}&\\bord3\\3a&H50&"
          f"\\fscx112\\fscy112\\t(0,{round((e - b) * 1000)},\\fscx100\\fscy100)", "27B", 1)
    ty.kicker(b, e, 3, "MTP ON SYCL", y=120)
    ty.slam(b, e, "27B, NOW FASTER.", 72, 92, 250, an=1)
    ramp, x, y = 1.0, 96, 440
    hit = b + ramp
    ty.streaks(b, hit + .6, 300, 600, seed=3)
    if fake:
        ty.slam(b, e, "##.#", 230, x, y, an=4)
    else:
        for frame in range(round(ramp * FPS)):
            p = frame / (ramp * FPS)
            ty.add(b + frame / FPS, b + (frame + 1) / FPS,
                  f"\\an4\\pos({x},{y})\\fs230\\b900\\fsp-6\\c&H{WHITE}&\\bord0\\blur.6", f"{m['tps'] * (1 - (1 - p) ** 4):.1f}")
        ty.add(hit, e, f"\\an4\\pos({x},{y})\\fs230\\b900\\fsp-6\\c&H{BLUE}&\\bord0\\blur.6\\fscx118\\fscy118"
              f"\\t(0,160,\\fscx100\\fscy100\\c&H{WHITE}&)", f"{m['tps']:.1f}")
        ty.add(hit, e, f"\\an4\\pos({x},{y})\\fs230\\b900\\fsp-6\\1a&HFF&\\3c&H{DEEP}&\\bord16\\blur18", f"{m['tps']:.1f}", 4)
    ty.ring(hit, 300, y)
    flashes.append(hit)
    shakes.append(hit)
    ty.label(hit, e, "TOKENS / SEC  ·  27B", 30, x + 6, y + 140, an=4, color=BLUE, spacing=10)
    if m.get("speculative_active") is True and m.get("speculative_evidence"):
        ty.slam(hit + .3, e, f"~{WINS['mtp']}× FASTER DECODE", 56, 92, 700, an=1, color=BLUE, glow=False)
        ty.label(hit + .3, e, "VS PLAIN DECODE ON A B60", 22, 100, 750, an=4, color=BLUE, spacing=5)
        rate = draft_rate(m["speculative_evidence"])
        if rate is not None:
            ty.label(hit + .3, e, f"{rate}% OF DRAFT TOKENS ACCEPTED", 22, 100, 790, an=4, alpha="40", spacing=5)
    # 5. Right GPU: the terminal's device line, held, with the caption on a dark patch below it.
    t = marks["terminal"]
    dev = f"Detected device (live app): {setup.get('gpu', {}).get('name')} · {setup.get('gpu', {}).get('vram_mb')} MiB"
    zb, z1 = t.get("zoom_box"), max(1.2, min(2.45, W / (max(len(t.get("card_line") or ""), len(dev)) * 14.5 + 96)))
    line_focus = (zb["x"] + W / (2 * z1), zb["y"] + zb["h"] + 270 / z1) if isinstance(zb, dict) and "x" in zb else (330, 250)
    # Half a second of the settled terminal (fully printed 1.6 s before the browser opens), slowed to fill
    # the shot, so the browser never shows. No HUD: the terminal's top line sits where it would go.
    b, e = clip("terminal", 4, max(0, t["browser"] - 1.2), speed=.25, zoom=(z1 - .2, z1), focus=line_focus)
    gpu_at = b
    ty.rect(b, e, 0, 640, W, 260, "0D0704", blur=24, layer=0)
    ty.rect(b + beat, e, 60, 206, 700, 4, BLUE, grow=240, layer=3)
    ty.kicker(b, e, 4, "RIGHT GPU", y=640)
    lines(b, e, "PICKS YOUR ARC.", "EVERY TIME.", 88)
    ty.label(b + beat, e, "EVEN WITH AN NVIDIA CARD IN THE SAME BOX", 20, W - 60, 820, an=6, alpha="30", spacing=4)
    # 6. Thinking budget: push into the settings panel as 256 goes in and saves.
    th = marks["thinking"]
    field = box_focus(th.get("budget_box"), (1180, 420))
    thinking_at = at(cursor)
    for i, (key, zoom) in enumerate((("ready", (1.25, 1.6)), ("saved", (1.6, 1.75)))):
        b, e = clip("thinking", 3, th[key], zoom=zoom, focus=field)
        if i == 0:  # the closer push-in puts the settings fields under the HUD
            hud(b, e)
        ty.rect(b, e, 0, H - 340, 760, 340, "000000", alpha="50", blur=70, layer=0)
        ty.kicker(b, e, 5, "THINKING BUDGET", y=640)
        ty.label(b, e, "NEW THINKING MODELS START AT 256", 20, 96, 590, an=4, alpha="30", spacing=4)
        lines(b, e, "CAP HOW LONG", "IT THINKS.", settle=i > 0)
    # 7. More cards recognised by name.
    b, e = clip(None, 8)
    cards_at = b
    ty.kicker(b, e, 6, "MORE CARDS", y=120)
    ty.slam(b, e, "MORE ARC PRO CARDS.", 96, 92, 250, an=1)
    for i, (name, note) in enumerate((("B50", "NEW  ·  16 GB"), ("B60", ""), ("B65", "NEW"), ("B70", "NEW"))):
        x, t0 = 96 + i * 320, b + (1 + i) * beat
        ty.rect(t0, e, x, 380, 290, 300, "1A120C" if note else "0E0A07", layer=1)
        ty.slam(t0, e, name, 120, x + 145, 510, color=BLUE if note else DIM, glow=bool(note))
        ty.label(t0, e, note, 22, x + 145, 630, color=BLUE, spacing=8)
        shakes.append(t0)
    ty.label(b + 5 * beat, e, "DETECTED BY NAME, VRAM MEASURED, NOT GUESSED", 22, 96, 760, an=4, alpha="30", spacing=4)
    # 8. End card.
    b, e = clip(None, 8)
    ty.slam(b, e, "ARC LLAMA", 176, W // 2, 330)
    ty.slam(b + beat, e, short, 132, W // 2, 500, color=BLUE)
    ty.rect(b + 2 * beat, e, 420, 590, 600, 76, "2A1A0C", blur=0, grow=180)
    ty.label(b + 2 * beat, e, "pip install -U arc-llama", 34, W // 2, 628, spacing=1)
    ty.label(b + 3 * beat, e, "NEW: arc-llama bench", 24, W // 2, 706, color=BLUE, spacing=3)
    ty.label(b + 3 * beat, e, "github.com/offbyonebit/arc-llama", 22, W // 2, 770, alpha="30", spacing=3)
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
    for name, seconds in (("open", at(3) + .3), ("startup", at(6 + 9) + .3), ("load", at(16 + 7) + .3),
                          ("mtp", mtp_at + 2.5), ("gpu", gpu_at + 1.8), ("thinking", thinking_at + 2.0),
                          ("cards", cards_at + 3.5), ("end-card", poster_at)):
        run("-ss", str(seconds), "-i", str(master), "-frames:v", "1", str(grabs / f"{name}.png"))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
