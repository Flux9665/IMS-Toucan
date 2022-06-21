import librosa.display as lbd
import matplotlib.pyplot as plt
import soundfile
import torch

from .InferenceArchitectures.InferenceFastSpeech2 import FastSpeech2
from .InferenceArchitectures.InferenceHiFiGAN import HiFiGANGenerator
from ..Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend
from ..Preprocessing.TextFrontend import get_language_id


class AnonFastSpeech2(torch.nn.Module):

    def __init__(self, device: str, path_to_hifigan_model: str, path_to_fastspeech_model: str):
        """
        Args:
            device: Device to run on. CPU is feasible, still faster than real-time, but a GPU is significantly faster.
            path_to_hifigan_model: Path to the vocoder model, including filename and suffix.
            path_to_fastspeech_model: Path to the synthesis model, including filename and suffix.

        """
        super().__init__()
        language = "en"
        self.device = device
        self.text2phone = ArticulatoryCombinedTextFrontend(language=language, add_silence_to_end=True)
        checkpoint = torch.load(path_to_fastspeech_model, map_location='cpu')
        self.phone2mel = FastSpeech2(weights=checkpoint["model"], lang_embs=None).to(torch.device(device))
        self.mel2wav = HiFiGANGenerator(path_to_weights=path_to_hifigan_model).to(torch.device(device))
        self.default_utterance_embedding = checkpoint["default_emb"].to(self.device)
        self.phone2mel.eval()
        self.mel2wav.eval()
        self.lang_id = get_language_id(language)
        self.to(torch.device(device))

    def forward(self, text, view=False, text_is_phonemes=False):
        """
        Args:
            text: The text that the TTS should convert to speech
            view: Boolean flag whether to produce and display a graphic showing the generated audio
            text_is_phonemes: Boolean flag whether the text parameter contains phonemes (True) or graphemes (False)

        Returns:
            48kHz waveform as 1d tensor

        """
        with torch.no_grad():
            phones = self.text2phone.string_to_tensor(text, input_phonemes=text_is_phonemes).to(torch.device(self.device))
            mel, durations, pitch, energy = self.phone2mel(phones,
                                                           return_duration_pitch_energy=True,
                                                           utterance_embedding=self.default_utterance_embedding)
            mel = mel.transpose(0, 1)
            wave = self.mel2wav(mel)
        if view:
            from IMSToucan.Utility.utils import cumsum_durations
            fig, ax = plt.subplots(nrows=2, ncols=1)
            ax[0].plot(wave.cpu().numpy())
            lbd.specshow(mel.cpu().numpy(),
                         ax=ax[1],
                         sr=16000,
                         cmap='GnBu',
                         y_axis='mel',
                         x_axis=None,
                         hop_length=256)
            ax[0].yaxis.set_visible(False)
            ax[1].yaxis.set_visible(False)
            duration_splits, label_positions = cumsum_durations(durations.cpu().numpy())
            ax[1].set_xticks(duration_splits, minor=True)
            ax[1].xaxis.grid(True, which='minor')
            ax[1].set_xticks(label_positions, minor=False)
            ax[1].set_xticklabels(self.text2phone.get_phone_string(text))
            ax[0].set_title(text)
            plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=.9, wspace=0.0, hspace=0.0)
            plt.show()
        return wave

    def anonymize_to_file(self, text: str, text_is_phonemes: bool, target_speaker_embedding: torch.tensor, path_to_result_file: str):
        """
        Args:
            text: The text that the TTS should convert to speech
            text_is_phonemes: Boolean flag whether the text parameter contains phonemes (True) or graphemes (False)
            target_speaker_embedding: The speaker embedding that should be used for the produced speech
            path_to_result_file: The path to the location where the resulting speech should be saved (including the filename and .wav suffix)

        """

        assert text.strip() != ""
        assert path_to_result_file.endswith(".wav")

        self.default_utterance_embedding = target_speaker_embedding.to(self.device)
        wav = self(text=text, text_is_phonemes=text_is_phonemes)
        soundfile.write(file=path_to_result_file, data=wav.cpu().numpy(), samplerate=48000)
