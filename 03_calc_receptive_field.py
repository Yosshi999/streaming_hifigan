from transformers import FastSpeech2ConformerTokenizer, FastSpeech2ConformerModel, FastSpeech2ConformerHifiGan
import soundfile as sf
import torch
import numpy as np

tokenizer = FastSpeech2ConformerTokenizer.from_pretrained("espnet/fastspeech2_conformer")
model = FastSpeech2ConformerModel.from_pretrained("espnet/fastspeech2_conformer")
hifigan = FastSpeech2ConformerHifiGan.from_pretrained("espnet/fastspeech2_conformer_hifigan")
sample_text = """O would some Power the gift to give us
To see ourselves as others see us!
It would from many a blunder free us,
And foolish notion"""

with torch.no_grad():
    inputs = tokenizer(sample_text, return_tensors="pt")
    input_ids = inputs["input_ids"]

    output_dict = model(input_ids, return_dict=True)
    spectrogram = output_dict["spectrogram"]

    # insert nan
    print("spec shape:", spectrogram.shape)
    spectrogram[0, 0, 0] = torch.nan
    waveform = hifigan(spectrogram).numpy()
    print(waveform.shape)
    get_rightmost_nan = lambda x: np.where(np.isnan(x))[0].max().item()

    rightmost_nan = get_rightmost_nan(waveform[0])
    assert np.all(np.isnan(waveform[0, :rightmost_nan+1]))  # all nan
    assert np.all(np.isfinite(waveform[0, rightmost_nan+1:]))  # all numeric
    # input[0] affects output[rightmost_nan]
    # input:  N..............
    #         ---#---
    #          --#--
    #           -#-
    #            #
    # output: NNNN...........
    print("receptive field is", rightmost_nan * 2 + 1)
    print("half receptive field is", rightmost_nan)

    log = []
    hid = spectrogram.transpose(2, 1)
    hid = hifigan.conv_pre(hid)
    log.append(get_rightmost_nan(hid[0, 0, :]))
    for i in range(hifigan.num_upsamples):
        hid = torch.nn.functional.leaky_relu(hid)
        hid = hifigan.upsampler[i](hid)
        log.append(get_rightmost_nan(hid[0, 0, :]))

        res = hifigan.resblocks[i * hifigan.num_kernels](hid)
        lo = [get_rightmost_nan(res[0, 0, :])]
        for j in range(1, hifigan.num_kernels):
            r = hifigan.resblocks[i * hifigan.num_kernels + j](hid)
            lo.append(get_rightmost_nan(r[0, 0, :]))
            res += r
        log.append(lo)
        hid = res / hifigan.num_kernels
    
    hid = torch.nn.functional.leaky_relu(hid)
    hid = hifigan.conv_post(hid)
    log.append(get_rightmost_nan(hid[0, 0, :]))
    hid = torch.tanh(hid)

    print(log)

    log2 = []
    # calc receptive field from nn
    x = hifigan.conv_pre.dilation[0] * (hifigan.conv_pre.kernel_size[0] // 2)
    log2.append(x)
    for i in range(hifigan.num_upsamples):
        assert isinstance(hifigan.upsampler[i], torch.nn.ConvTranspose1d)
        x *= hifigan.upsampler[i].stride[0]
        x += hifigan.upsampler[i].kernel_size[0] - 1 - hifigan.upsampler[i].padding[0]
        log2.append(x)
        ys = []
        for j in range(hifigan.num_kernels):
            y = 0
            resblock = hifigan.resblocks[i * hifigan.num_kernels + j]
            for conv1, conv2 in zip(resblock.convs1, resblock.convs2):
                y += conv1.dilation[0] * (conv1.kernel_size[0] // 2)
                y += conv2.dilation[0] * (conv2.kernel_size[0] // 2)
            ys.append(y)
        x += max(ys)
        log2.append(x)
    x += hifigan.conv_post.dilation[0] * (hifigan.conv_post.kernel_size[0] // 2)
    log2.append(x)
    print(log2)

    print("predicted half receptive field:", x)
