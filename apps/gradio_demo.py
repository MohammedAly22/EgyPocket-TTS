"""Streaming CPU demo for an EgyPocket-TTS bundle.

    pip install "gradio>=5"
    python apps/gradio_demo.py --bundle /path/to/egypocket_6l [--port 7860] [--share]

Audio is streamed to the browser while it is being generated, and the page
shows the time to first audio and the real-time factor, measured on the CPU.
"""

import argparse
import threading
import time

import gradio as gr
import numpy as np
import torch

from egypocket.inference import EgyPocketTTS

EXAMPLES = [
    ["النهارده الجو حلو قوي, وانا رايح الشغل بدري علشان الحق الاجتماع."],
    ["ولو ركزنا شوية في الحكاية دي, هنلاقي ان المصريين القدماء كانوا سابقين عصرهم في حاجات كتير. وده اللي هنتكلم عنه النهارده."],
    ["رحت المدينة امبارح."],
    ["رحت المدينةْ امبارح."],
    ["العِلْم نور, والعَلَم رمز البلد."],
    ["علشان اَلْحَق~ْ الاجتماع."],
    ["الاجتماع هيبدأ الساعة 3:30 وهيخلص على 5."],
]


def build_app(tts: EgyPocketTTS) -> gr.Blocks:
    lock = threading.Lock()  # one generation at a time: the model is not thread-safe
    sr = tts.sample_rate

    def stats_md(first: float | None, audio_sec: float, wall: float, done: bool) -> str:
        rtf = audio_sec / wall if wall > 0 else 0.0
        state = "done" if done else "streaming..."
        first_txt = f"{first * 1000:.0f} ms" if first is not None else "-"
        return (
            f"**{state}** | first audio: **{first_txt}** | audio: {audio_sec:.2f} s | "
            f"compute: {wall:.2f} s | speed: **x{rtf:.2f} real time**"
        )

    def synthesize(text: str, voice: str | None, threads: int, min_chunk_sec: float):
        if not text or not text.strip():
            raise gr.Error("Write some Egyptian Arabic text first.")
        with lock:
            torch.set_num_threads(int(threads))
            try:
                normalized = tts.normalize(text)
            except Exception as exc:  # noqa: BLE001 - shown to the user
                raise gr.Error(str(exc)) from exc
            t0 = time.monotonic()
            first = None
            buffer: list[np.ndarray] = []
            buffered = 0
            full: list[np.ndarray] = []
            produced = 0
            # The first chunk goes out as soon as possible; later ones are grouped
            # so the browser is not flooded with 80 ms pieces.
            target = int(0.24 * sr)
            for chunk in tts.stream(text, voice or None):
                buffer.append(chunk)
                buffered += len(chunk)
                full.append(chunk)
                produced += len(chunk)
                if buffered >= target:
                    if first is None:
                        first = time.monotonic() - t0
                    pcm = np.concatenate(buffer)
                    buffer, buffered = [], 0
                    target = int(min_chunk_sec * sr)
                    yield (
                        (sr, (np.clip(pcm, -1, 1) * 32767).astype(np.int16)),
                        stats_md(first, produced / sr, time.monotonic() - t0, False),
                        normalized,
                        gr.skip(),
                    )
            wall = time.monotonic() - t0
            if buffer:
                if first is None:
                    first = wall
                pcm = np.concatenate(buffer)
                yield (
                    (sr, (np.clip(pcm, -1, 1) * 32767).astype(np.int16)),
                    stats_md(first, produced / sr, wall, False),
                    normalized,
                    gr.skip(),
                )
            audio = np.concatenate(full) if full else np.zeros(1, dtype=np.float32)
            yield (
                gr.skip(),
                stats_md(first, produced / sr, wall, True),
                normalized,
                (sr, (np.clip(audio, -1, 1) * 32767).astype(np.int16)),
            )

    with gr.Blocks(title="EgyPocket-TTS") as app:
        gr.Markdown(
            "# EgyPocket-TTS: streaming on CPU\n"
            "Egyptian Arabic, 6-layer student (~105M parameters), running on this machine's CPU. "
            "Write diacritics only where a word is read wrongly (e.g. `المدينةْ`, `العَلَم`, `ق~َوِي`)."
        )
        with gr.Row():
            with gr.Column(scale=3):
                text = gr.Textbox(label="Text", lines=4, rtl=True, value=EXAMPLES[0][0])
                with gr.Row():
                    threads = gr.Slider(1, 8, value=2, step=1, label="CPU threads")
                    min_chunk = gr.Slider(
                        0.1, 1.0, value=0.4, step=0.1, label="Streaming chunk size (s)"
                    )
                voice = gr.Audio(
                    label="Voice prompt (optional, default = bundle voice)",
                    type="filepath",
                    sources=["upload", "microphone"],
                )
                button = gr.Button("Speak", variant="primary")
            with gr.Column(scale=3):
                stats = gr.Markdown("")
                live = gr.Audio(label="Live stream", streaming=True, autoplay=True, format="wav")
                normalized = gr.Textbox(label="What the model reads (normalized)", rtl=True)
                full = gr.Audio(label="Full audio (replay / download)", type="numpy")
        gr.Examples(EXAMPLES, inputs=[text])
        button.click(
            synthesize, inputs=[text, voice, threads, min_chunk], outputs=[live, stats, normalized, full]
        )
    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True, help="exported bundle folder (export/egypocket_6l)")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--share", action="store_true", help="public gradio.live link")
    a = p.parse_args()
    tts = EgyPocketTTS(a.bundle, num_threads=2)
    tts.generate("اهلا.")  # warm-up so the first request shows steady-state latency
    build_app(tts).queue().launch(server_name=a.host, server_port=a.port, share=a.share)


if __name__ == "__main__":
    main()
