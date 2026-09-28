# EgyPocket-TTS

A fast, lightweight **Egyptian-Arabic** text-to-speech model that runs on a **CPU faster than real time**, built on Kyutai's [Pocket TTS](https://github.com/kyutai-labs/pocket-tts) ([CALM paper](https://arxiv.org/abs/2509.06926)). It reads plain, undiacritized Egyptian Arabic, and it lets you **override pronunciation with diacritics** wherever it reads a word wrongly, including control over final **ة**.

* 6-layer, ~100M-parameter student (the size of the released Pocket TTS): one transformer pass plus a 1-step flow head per 80 ms frame, streaming Mimi decoding.
* Trained on [Masri-100h](https://huggingface.co/datasets/ehabnegm/100-hour-Egyptian-dataset-single-speaker) (single narrator, ~60 h after filtering).
* CATT (Egyptian diacritizer) is used **only to prepare training data**. It is not part of inference.

---

## How it works

### Model: Pocket TTS, adapted to Egyptian Arabic

Pocket TTS is a continuous audio language model. A causal transformer reads `[voice prompt latents | text tokens | audio latents so far]`. For every 80 ms frame, a small MLP head trained with Lagrangian Self-Distillation (LSD) samples the next 32-dim Mimi latent in **one step**. The Mimi codec turns latents into 24 kHz audio in a parallel streaming thread. Training follows Kyutai's released two-stage recipe:

1. **Teacher (24 layers)**: start from Pocket TTS's English 24-layer teacher, replace the text embedding with a new one for our tokenizer, and fine-tune everything (`configs/teacher_24l.yaml`, which is pocket-tts's `finetune_language` recipe). The acoustic knowledge (Mimi latent space, prosody, voice conditioning) transfers; the model learns to read Egyptian Arabic.
2. **Student (6 layers)**: depth distillation of the teacher with classifier-free guidance 2.0 baked in (`configs/distill_6l.yaml`, pocket-tts's `depth_distill` recipe). The student seeds its layers from the teacher's bottom and top layers and copies the text embedding, voice projection and sampler head. At inference it needs one 6-layer pass per frame. This is the CPU model.

### Text: diacritics are pronunciation control, not decoration

* **Character tokenizer** (`egypocket/text/tokenizer.py`). Every letter and every diacritic is its own token, so a diacritized word shares all its letter tokens with the plain spelling and the marks are purely additive information.
* **ة is explicit.** `ة` followed by a vowel mark is one token (`ةْ ةَ ةِ ةُ ةً ةٌ ةٍ`), so `مَدِينَة` (… *-a*) and `مَدِينَةْ` (… *-at*) differ by exactly one dedicated symbol. The five forms `ة ةْ ةَ ةِ ةُ` never collapse to one representation (unit-tested).
* **One frontend for training and inference** (`egypocket/text/frontend.py`). It applies NFKC, verbalizes Egyptian numbers (`1994` → `الف وتسعمية واربعة وتسعين`), and unifies alef (أ إ آ → ا, as in the corpus). It puts each letter's marks in the canonical order `[~][shadda][vowel]`, drops orphan marks, and reduces punctuation to `,` (pause) and `.` (sentence end).
* **CATT output cleanup** (`normalize_catt_output` in `egypocket/text/catt.py`): `^ < >` are removed, `؞` becomes `~` (an Egyptian pronunciation marker, e.g. ق read as a glottal stop), and diacritics are preserved. It is covered by unit tests.

### Data: why the preparation has ten steps

* **The corpus writes every ة as ه** (`حاجه`, `المدينه`). 19% of all words end in ه, and the model would never see ة. So CATT is run once on the raw text. For each word type ending in ه, CATT's vowel on the previous letter votes: fatha → taa marbuta (`حَاجَه`), damma → pronoun (`كِتَابُه`). Types with ≥ 60% ة votes are restored everywhere, and the list is saved as a lexicon that the inference frontend applies too. Then CATT runs again on the restored text, which it reads better (`ورحمة` → `وَرَحْمَةُ` rather than the verb `وَرَحِمَهُ`). Those diacritics are what the model learns from.
* **Diacritic mixing.** Each training sample is plain (55%), partially diacritized (30%, 10–50% of words), or fully diacritized (15%). A diacritized word sometimes keeps only a random subset of its marks, so a single mark (e.g. only `ةْ`) is also meaningful. The model stays a strong plain-text reader, and diacritics work as an override.
* **Numbers.** 4,971 clips (24 h) contain digits. They are verbalized, and clips whose text does not match the audio are removed by the alignment-score filter.
* **Word alignment** (wav2vec2 CTC). Pocket TTS trains by cutting each utterance at a word boundary: the audio before the cut is the voice prompt, and the audio after it is the target. Boundaries are snapped into measured pauses. The corpus has no punctuation, so pauses become `,` / `.`.
* **Everything is cached.** Pocket's own trainer re-encodes audio with Mimi at every step. Here the full-utterance latents and the cold-start latents at every possible cut are precomputed, so a training step reads only in-RAM numpy arrays: no audio decoding, no resampling, no encoder on the GPU.

### Monitoring

TensorBoard (`runs/<name>/tb`) shows:
* train and validation losses, learning rate, gradient norm, steps/s, audio-seconds/s and GPU memory;
* every 1000 steps, **audio samples** of 5 fixed sentences, generated with the EMA weights that get exported. They include the override pairs `رحت المدينة` / `رحت المدينةْ` and `العِلْم` / `العَلَم`;
* at step 0, the Mimi resynthesis of a validation clip, which is the quality ceiling.

---

## Repository layout

```
configs/                  teacher_24l.yaml, distill_6l.yaml (training recipes)
notebooks/                01_prepare_data → 02_train_teacher → 03_distill_student → 04_inference
scripts/setup_runpod.sh   conda env + dependencies + Jupyter kernel
egypocket/
  text/                   frontend, numbers, CATT integration, ة restoration, tokenizer
  data/                   manifests, CATT passes, alignment, latent cache, batch loader
  training/               trainer (TensorBoard, EMA, checkpoints) + adapted pocket-tts modules
  modelcfg.py             pocket-tts model configs for the teacher/student
  bundle.py               export a run to a self-contained CPU bundle
  inference.py            EgyPocketTTS: frontend + chunking + pocket-tts streaming engine
  cli.py                  every step as a command
tests/                    CATT normalization, frontend, ة restoration, tokenizer
```

---

## Training on RunPod: step by step

### 0. Before you start

* **Kyutai weights (gated).** Log in at https://huggingface.co/kyutai/pocket-tts and click to accept the terms. Then create a *read* token at https://huggingface.co/settings/tokens. The ungated release has an all-zero Mimi encoder and cannot be used for training.
* **CATT.** Zip your `catt_tashkeel` folder (it must contain the `.py` files and `checkpoints/eca_model_weights.pt`). CATT's own conda env is **not** needed. It runs inside the `egypocket` env, which only needs torch and pytorch_lightning; the setup script installs both.

### 1. Create the pod

* GPU: **1 × RTX 4090 (24 GB)**.
* Template: any RunPod PyTorch / CUDA 12 template with JupyterLab (e.g. "RunPod Pytorch 2.x").
* **Volume disk: 100 GB**, mounted at `/workspace`. Everything goes there: env, data (~15 GB), models, checkpoints (~15 GB).
* Container disk: 30 GB.
* **Expose HTTP ports: `8888,6006`**. 8888 is JupyterLab and 6006 is TensorBoard.

### 2. Clone and install (JupyterLab → Terminal)

```bash
cd /workspace
git clone https://github.com/MohammedAly22/EgyPocket-TTS.git
bash EgyPocket-TTS/scripts/setup_runpod.sh
```

This takes about 10 minutes. It installs Miniforge into `/workspace/miniforge3`, creates the conda env `egypocket` (Python 3.11, torch 2.14.0 + CUDA 12.6), installs the project, and registers the Jupyter kernel **"Python (egypocket)"**.

### 3. Upload CATT

In the JupyterLab file browser, go to `/workspace`, click the upload button and choose `catt_tashkeel.zip`. Then in the terminal:

```bash
cd /workspace && unzip -q catt_tashkeel.zip && ls /workspace/catt_tashkeel/checkpoints
# must list: eca_model_weights.pt
```

(From your PC you can also use `runpodctl send catt_tashkeel.zip` and `runpodctl receive <code>` on the pod.)

### 4. Prepare the data (about 1 hour)

Open `EgyPocket-TTS/notebooks/01_prepare_data.ipynb`, select the kernel **Python (egypocket)**, and run the cells from top to bottom. The login cell asks for your HF token. The notebook shows the results of every step: stats, CATT raw vs cleaned output, the ة report, tokenization, the alignment-score histogram, and decoded training samples you can listen to.

### 5. Train the teacher (`02_train_teacher.ipynb`)

Run the cells. The TensorBoard cell prints its URL (`https://<pod-id>-6006.proxy.runpod.net`), and the launch cell starts training **in the background**. You can close the browser; re-open the notebook later and run the monitoring cells (status, inline plots, audio samples). Stop with the STOP cell, which saves a checkpoint of the current step first. Launching again resumes from the newest checkpoint.

Expect roughly 1.2–2 s per step on a 4090 (effective batch 64); the exact speed is shown live as `optim/steps_per_sec`. The default budget is 30k steps. Listen to the samples: once the sentences are correct, the ة / diacritic pairs sound different, and `valid/flow_loss` is flat, you can stop earlier.

### 6. Distill the CPU student (`03_distill_student.ipynb`)

It uses the newest teacher checkpoint by default (or pick one) and trains the 6-layer student. The default is 20k steps.

### 7. Export and run on CPU (`04_inference.ipynb`)

This exports `export/egypocket_6l/`, runs it on 2 CPU threads, and prints the real-time factor and first-audio latency. It then demonstrates the diacritic overrides, streaming, voice prompts, int8 quantization and held-out test sentences, and zips the bundle for download.

### The same pipeline from a terminal (optional, e.g. inside `tmux`)

```bash
source /workspace/miniforge3/etc/profile.d/conda.sh && conda activate egypocket
cd /workspace/EgyPocket-TTS
export EGY_DATA=/workspace/egypocket_data HF_HOME=/workspace/hf_cache
huggingface-cli login                                        # paste the token once
python -m egypocket.cli prepare --catt-dir /workspace/catt_tashkeel
python -m pytest -q tests
python -m egypocket.training.train configs/teacher_24l.yaml
TEACHER_CKPT=$(ls $EGY_DATA/runs/teacher_24l/checkpoint_*.pt | tail -1) \
  python -m egypocket.training.train configs/distill_6l.yaml
python -m egypocket.cli export --run $EGY_DATA/runs/distill_6l --name egypocket_6l --layers 6
python -m egypocket.cli say --bundle $EGY_DATA/export/egypocket_6l --threads 2 \
  --text "النهارده الجو حلو قوي." --out out.wav
tensorboard --logdir $EGY_DATA/runs --port 6006 --bind_all   # in another terminal
```

---

## Using the model

```python
from egypocket.inference import EgyPocketTTS

tts = EgyPocketTTS("export/egypocket_6l", num_threads=2)
audio = tts.generate("النهارده الجو حلو قوي, وانا رايح الشغل بدري.")   # float32, 24 kHz
tts.save("رحت المدينةْ امبارح.", "override.wav")                      # ةْ -> ...-at
for chunk in tts.stream("نص طويل ..."):                                # streaming
    ...
print(tts.last_stats.real_time_factor, tts.last_stats.first_chunk_sec)
```

**Overriding pronunciation.** Write diacritics only on the word, or even the single letter, that is read wrongly:

| input | reading |
|---|---|
| `مدينة` | model decides: pausal *-a*, or *-it* in a construct |
| `مدينةْ` / `مدينةَ` / `مدينةِ` / `مدينةُ` | the *t* is pronounced (with that vowel) |
| `العِلْم` vs `العَلَم` | *il-ʿilm* vs *il-ʿalam* |
| `ق~` | Egyptian marker learned from CATT's `؞` (e.g. ق read as a glottal stop) |

Punctuation: `,` / `،` produce a pause, and sentence ends (`. ! ? ؟`) produce a full stop. Latin script is dropped (with a warning); code-switching is future work.

---

## Scaling up (next experiments)

* **More speakers / zero-shot cloning.** The architecture already conditions on a voice prompt; cloning quality comes from speaker diversity. Add datasets as extra manifests (same pipeline), rebuild the caches, and continue from the teacher checkpoint. The CPU speed is unchanged because the student size is unchanged.
* **Code-switching.** Add Latin letters (or an English phoneme set) to `egypocket/text/tokenizer.py` and to the frontend, and train with mixed data. The embedding table grows by a few rows; nothing else changes.

## Troubleshooting

* `ZERO ENCODER` / 401 errors: the terms at https://huggingface.co/kyutai/pocket-tts were not accepted, or no token is set.
* Out of GPU memory: set `batch_size: 4` and `grad_accum_steps: 16` in the config (effective batch stays 64).
* `torch.compile` errors on an unusual driver/image: set `compile: false` in the config (about 20% slower, otherwise identical).
* `non-finite gradient`: run the launch cell again; training resumes from the last checkpoint.
* Pod restarted: re-run `bash scripts/setup_runpod.sh` (fast, only re-registers the kernel), then re-launch; training resumes automatically.

Licences and credits: see [NOTICE.md](NOTICE.md). The dataset is CC BY-NC 4.0 and features a real person's voice: use it for research, and label synthetic audio.
