# Third-party code and models

## Pocket TTS (Kyutai) — MIT licence

EgyPocket-TTS runs on the `pocket-tts` package (installed from
https://github.com/kyutai-labs/pocket-tts, pinned commit
`797209501de994aaa4e84a4e29f2fd98d2f5283a`) and starts training from Kyutai's
released Pocket TTS weights (https://huggingface.co/kyutai/pocket-tts; accept
their terms before use).

The following files are adapted from the pocket-tts `training/` directory:
`egypocket/training/{model,conditioner,samplers,utils,checkpointing,distributed,builders,shrink}.py`,
and parts of `egypocket/training/train.py`, `egypocket/training/args.py`,
`egypocket/data/align.py` (batched CTC Viterbi) and `egypocket/data/cache.py`
(stitch calibration, Mimi hash).

```
Copyright (c) Kyutai

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

## CATT Egyptian diacritizer

Used for data preparation only (not shipped, not used at inference). Egyptian
number verbalization in `egypocket/text/numbers.py` is adapted from CATT's
`ECAPreProcessor`.

## Data

Masri 100h (https://huggingface.co/datasets/ehabnegm/100-hour-Egyptian-dataset-single-speaker),
CC BY-NC 4.0 — research / non-commercial use; the voice belongs to a real
person, see the dataset card before publishing anything generated with it.

## Alignment model

`jonatasgrosman/wav2vec2-large-xlsr-53-arabic` (Apache-2.0) for forced alignment.
