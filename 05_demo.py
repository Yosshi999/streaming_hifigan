from transformers import FastSpeech2ConformerTokenizer, FastSpeech2ConformerModel, FastSpeech2ConformerHifiGan
import torch
import onnxruntime
import numpy as np
from numpy.lib.stride_tricks import as_strided
import sounddevice

import argparse
from collections import deque
import math
import time

parser = argparse.ArgumentParser()
parser.add_argument("--chunked", action="store_true", help="Use chunked version for faster latency")
parser.add_argument("--output", type=int, help="audio output device ID (See python -m sounddevice for a list of devices)")
args = parser.parse_args()

HIFI_SAMPLE_RATE = 22050
IN_MARGIN = math.ceil(3513 / 256)
OUT_MARGIN = IN_MARGIN * 256

VALID_IN_CHUNK = (HIFI_SAMPLE_RATE * 1) // 256
VALID_OUT_CHUNK = VALID_IN_CHUNK * 256

IN_CHUNK = IN_MARGIN * 2 + VALID_IN_CHUNK
IN_STRIDE = VALID_IN_CHUNK

tokenizer = FastSpeech2ConformerTokenizer.from_pretrained("espnet/fastspeech2_conformer")
model = FastSpeech2ConformerModel.from_pretrained("espnet/fastspeech2_conformer")
session = onnxruntime.InferenceSession("hifigan.onnx")

print(model.device)

outq = deque()
def process(outdata: np.ndarray, frames, time, status):
    """This is called from a separate thread for each audio block."""
    outdata.fill(0)
    iteration = 0
    while len(outq) > 0 and iteration < len(outdata):
        data = outq.popleft()
        data_send = data[:len(outdata) - iteration]
        outdata[iteration : iteration + len(data_send), 0] = data_send
        iteration += len(data_send)
        if len(data) > len(data_send):
            outq.appendleft(data[len(data_send):])

def tts_full(sample_text: str) -> np.ndarray:
    with torch.no_grad():
        inputs = tokenizer(sample_text, return_tensors="pt")
        input_ids = inputs["input_ids"]

        output_dict = model(input_ids, return_dict=True)
        spectrogram = output_dict["spectrogram"].numpy()

    waveform_full = session.run(["wave"], {"spectrogram": spectrogram})[0]
    return waveform_full.ravel()

def tts_chunked(sample_text: str):
    with torch.no_grad():
        inputs = tokenizer(sample_text, return_tensors="pt")
        input_ids = inputs["input_ids"]

        output_dict = model(input_ids, return_dict=True)
        spectrogram = output_dict["spectrogram"].numpy()
    spec = spectrogram[0]  # (L, dim)
    inlen = spec.shape[0]
    nchunk = math.ceil(max(0, inlen - IN_MARGIN * 2) / VALID_IN_CHUNK)
    inlen_with_pad = nchunk * VALID_IN_CHUNK + IN_MARGIN * 2
    pad = inlen_with_pad - inlen
    assert pad >= 0
    spec = np.pad(spec, ((0, pad), (0, 0)))
    specview = as_strided(spec, (nchunk, IN_CHUNK, spec.shape[1]), (spec.strides[0] * VALID_IN_CHUNK, spec.strides[0], spec.strides[1]))
    for i, view in enumerate(specview):
        segm = session.run(["wave"], {"spectrogram": view[None]})[0][0]  # (Lout,)
        if len(specview) == 1:
            # only 1 chunk
            yield segm[:len(segm)-pad*256]
        elif i == 0:
            # first chunk
            yield segm[:-OUT_MARGIN]
        elif i == len(specview) - 1:
            # last chunk
            yield segm[OUT_MARGIN:len(segm)-pad*256]
        else:
            yield segm[OUT_MARGIN:-OUT_MARGIN]

try:
    with sounddevice.OutputStream(
        samplerate = HIFI_SAMPLE_RATE,
        device = args.output,
        dtype = np.float32,
        channels = 1,
        callback = process,
    ):
        while True:
            text = input("> ")
            if args.chunked:
                for chunk in tts_chunked(text):
                    outq.append(chunk)
            else:
                outq.append(tts_full(text))


except KeyboardInterrupt:
    print("\nInterrupted by user. Exiting...")