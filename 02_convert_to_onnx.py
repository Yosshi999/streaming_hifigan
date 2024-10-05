from transformers import FastSpeech2ConformerTokenizer, FastSpeech2ConformerModel, FastSpeech2ConformerHifiGan
import torch
import onnxruntime
import numpy as np
import time

tokenizer = FastSpeech2ConformerTokenizer.from_pretrained("espnet/fastspeech2_conformer")
model = FastSpeech2ConformerModel.from_pretrained("espnet/fastspeech2_conformer")
hifigan = FastSpeech2ConformerHifiGan.from_pretrained("espnet/fastspeech2_conformer_hifigan")

with torch.no_grad():
    inputs = tokenizer("Hello, my dog is cute.", return_tensors="pt")
    input_ids = inputs["input_ids"]

    output_dict = model(input_ids, return_dict=True)
    spectrogram = output_dict["spectrogram"]


    torch.onnx.export(
        hifigan,
        spectrogram,
        "hifigan.onnx",
        verbose=True,
        input_names=["spectrogram"],
        output_names=["wave"],
        dynamic_axes={
            "spectrogram": {
                0: "batch",
                1: "length",
            },
            "wave": {
                0: "batch",
                1: "outlength",
            },
        })

    # verification
    waveform_expected = hifigan(spectrogram).numpy()

    session = onnxruntime.InferenceSession("hifigan.onnx")
    waveform_actual = session.run(["wave"], {"spectrogram": spectrogram.numpy()})[0]

    print("expected:", waveform_expected)
    print("actual:", waveform_actual)
    assert waveform_expected.shape == waveform_actual.shape
    print("max diff:", np.abs(waveform_actual - waveform_expected).max())

    print("compare performance...")
    sample_text = """O would some Power the gift to give us
    To see ourselves as others see us!
    It would from many a blunder free us,
    And foolish notion"""

    t0 = time.perf_counter()
    inputs = tokenizer(sample_text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    t1 = time.perf_counter()
    output_dict = model(input_ids, return_dict=True)
    spectrogram = output_dict["spectrogram"]
    t2 = time.perf_counter()
    waveform = hifigan(spectrogram)
    t3 = time.perf_counter()
    waveform = session.run(["wave"], {"spectrogram": spectrogram.numpy()})[0]
    t4 = time.perf_counter()
    print("## processing time ##")
    print(f"tokenize: {t1 - t0 : .3f}s")
    print(f"spectro:  {t2 - t1 : .3f}s")
    print(f"torch hifigan:  {t3 - t2 : .3f}s")
    print(f"onnx  hifigan:  {t4 - t3 : .3f}s")