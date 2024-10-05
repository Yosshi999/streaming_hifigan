from transformers import FastSpeech2ConformerTokenizer, FastSpeech2ConformerModel, FastSpeech2ConformerHifiGan
import torch
import onnxruntime
import numpy as np
from numpy.lib.stride_tricks import as_strided
import soundfile as sf
import matplotlib.pyplot as plt
import math
import time

IN_MARGIN = math.ceil(3513 / 256)
OUT_MARGIN = IN_MARGIN * 256

VALID_IN_CHUNK = (22050 * 1) // 256
VALID_OUT_CHUNK = VALID_IN_CHUNK * 256

IN_CHUNK = IN_MARGIN * 2 + VALID_IN_CHUNK
IN_STRIDE = VALID_IN_CHUNK


tokenizer = FastSpeech2ConformerTokenizer.from_pretrained("espnet/fastspeech2_conformer")
model = FastSpeech2ConformerModel.from_pretrained("espnet/fastspeech2_conformer")
session = onnxruntime.InferenceSession("hifigan.onnx")

print(model.device)
warmup_text = "warming up"
with torch.no_grad():
    inputs = tokenizer(warmup_text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    output_dict = model(input_ids, return_dict=True)
    spectrogram = output_dict["spectrogram"]
waveform_actual = session.run(["wave"], {"spectrogram": spectrogram.numpy()})[0]

sample_text = """O would some Power the gift to give us
To see ourselves as others see us!
It would from many a blunder free us,
And foolish notion"""

with torch.no_grad():
    t0 = time.perf_counter()
    inputs = tokenizer(sample_text, return_tensors="pt")
    input_ids = inputs["input_ids"]

    t1 = time.perf_counter()
    output_dict = model(input_ids, return_dict=True)
    spectrogram = output_dict["spectrogram"]

    t2 = time.perf_counter()

waveform_full = session.run(["wave"], {"spectrogram": spectrogram.numpy()})[0]
t3 = time.perf_counter()

spec = spectrogram.numpy()[0]  # (L, dim)
inlen = spec.shape[0]
nchunk = math.ceil(max(0, inlen - IN_MARGIN * 2) / VALID_IN_CHUNK)
inlen_with_pad = nchunk * VALID_IN_CHUNK + IN_MARGIN * 2
pad = inlen_with_pad - inlen
assert pad >= 0
spec = np.pad(spec, ((0, pad), (0, 0)), "constant", constant_values=0)
specview = as_strided(spec, (nchunk, IN_CHUNK, spec.shape[1]), (spec.strides[0] * VALID_IN_CHUNK, spec.strides[0], spec.strides[1]))

outs = []
for i, view in enumerate(specview):
    if len(specview) == 1:
        # only 1 chunk
        segm = session.run(["wave"], {"spectrogram": view[None, :len(view)-pad]})[0][0]  # (Lout,)
        outs.append(segm)
        t4 = time.perf_counter()
    elif i == 0:
        # first chunk
        segm = session.run(["wave"], {"spectrogram": view[None, :]})[0][0]  # (Lout,)
        outs.append(segm[:-OUT_MARGIN])
        t4 = time.perf_counter()
    elif i == len(specview) - 1:
        # last chunk
        segm = session.run(["wave"], {"spectrogram": view[None, :len(view)-pad]})[0][0]  # (Lout,)
        outs.append(segm[OUT_MARGIN:])
    else:
        segm = session.run(["wave"], {"spectrogram": view[None, :]})[0][0]  # (Lout,)
        outs.append(segm[OUT_MARGIN:-OUT_MARGIN])
outwave = np.concatenate(outs)
assert len(outwave) == inlen * 256
t5 = time.perf_counter()

sf.write("speech_full.wav", waveform_full.ravel(), samplerate=22050)
sf.write("speech_chunked.wav", outwave, samplerate=22050)
diff = np.abs(waveform_full.ravel() - outwave)
plt.plot(np.arange(len(diff)), diff, "r-")
plt.xlabel("frame")
plt.ylabel("abs diff")
plt.savefig("diff.png")
print("diff max:", diff.max())
print("diff med:", np.median(diff))
print("diff mean:", np.mean(diff))

print("## processing time ##")
print(f"tokenize: {t1 - t0 : .3f}s")
print(f"spectro:  {t2 - t1 : .3f}s")
print(f"hifigan:  {t3 - t2 : .3f}s")
print(f"chunked hifigan (latency):  {t4 - t3 : .3f}s")
print(f"chunked hifigan (all):  {t5 - t3 : .3f}s")