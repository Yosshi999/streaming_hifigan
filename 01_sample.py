from transformers import FastSpeech2ConformerTokenizer, FastSpeech2ConformerModel, FastSpeech2ConformerHifiGan
import soundfile as sf
import torch
import time

tokenizer = FastSpeech2ConformerTokenizer.from_pretrained("espnet/fastspeech2_conformer")
model = FastSpeech2ConformerModel.from_pretrained("espnet/fastspeech2_conformer")
hifigan = FastSpeech2ConformerHifiGan.from_pretrained("espnet/fastspeech2_conformer_hifigan")

print(model.device)
warmup_text = "warming up"

inputs = tokenizer(warmup_text, return_tensors="pt")
input_ids = inputs["input_ids"]
output_dict = model(input_ids, return_dict=True)
spectrogram = output_dict["spectrogram"]
waveform = hifigan(spectrogram)
print("input:", warmup_text)
print("tokens:", input_ids.shape)
print("spectrogram:", spectrogram.shape)
print("waveform:", waveform.shape)


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
    waveform = hifigan(spectrogram)

    t3 = time.perf_counter()
    sf.write("speech.wav", waveform.squeeze().detach().numpy(), samplerate=22050)

print("input:", sample_text)
print("tokens:", input_ids.shape)
print("spectrogram:", spectrogram.shape)
print("waveform:", waveform.shape)

print("## processing time ##")
print(f"tokenize: {t1 - t0 : .3f}s")
print(f"spectro:  {t2 - t1 : .3f}s")
print(f"hifigan:  {t3 - t2 : .3f}s")