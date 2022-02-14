import gradio as gr
import numpy as np
import torch

from InferenceInterfaces.Meta_FastSpeech2 import Meta_FastSpeech2


def float2pcm(sig, dtype='int16'):
    """
    https://gist.github.com/HudsonHuang/fbdf8e9af7993fe2a91620d3fb86a182
    """
    sig = np.asarray(sig)
    if sig.dtype.kind != 'f':
        raise TypeError("'sig' must be a float array")
    dtype = np.dtype(dtype)
    if dtype.kind not in 'iu':
        raise TypeError("'dtype' must be an integer type")
    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)


class TTS_Interface:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Meta_FastSpeech2(device=self.device)

    def read(self, prompt, language):
        language_id_lookup = {
            "English"  : "en",
            "German"   : "de",
            "Greek"    : "el",
            "Spanish"  : "es",
            "Finnish"  : "fi",
            "Russian"  : "ru",
            "Hungarian": "hu",
            "Dutch"    : "nl",
            "French"   : "fr"
            }
        self.model.set_language(language_id_lookup[language])
        wav = self.model(prompt)
        return 48000, float2pcm(wav.cpu().numpy())


meta_model = TTS_Interface()

iface = gr.Interface(fn=meta_model.read,
                     inputs=[gr.inputs.Textbox(lines=2, placeholder="write what you want the synthesis to read here...", label=" "),
                             gr.inputs.Dropdown(['English',
                                                 'German',
                                                 'Greek',
                                                 'Spanish',
                                                 'Finnish',
                                                 'Russian',
                                                 'Hungarian',
                                                 'Dutch',
                                                 'French'], type="value", default='English', label="Language Selection")],
                     outputs=gr.outputs.Audio(type="numpy", label=None),
                     layout="vertical",
                     title="IMS Toucan Multilingual Multispeaker Demo",
                     thumbnail="Utility/toucan.png",
                     theme="default",
                     allow_flagging="never",
                     allow_screenshot=False)
iface.launch()
