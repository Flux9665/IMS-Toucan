import soundfile as sf
import torch
import torch.multiprocessing
import torch.multiprocessing
from numpy import trim_zeros
from speechbrain.pretrained import EncoderClassifier

from Preprocessing.AudioPreprocessor import AudioPreprocessor
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.EnergyCalculator import EnergyCalculator
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.PitchCalculator import Dio


class ProsodicConditionExtractor:

    def __init__(self, sr, device=torch.device("cpu")):
        self.dio = Dio(reduction_factor=1, fs=16000, use_token_averaged_f0=False)
        self.energy_calc = EnergyCalculator(reduction_factor=1, fs=16000, use_token_averaged_energy=False)
        self.ap = AudioPreprocessor(input_sr=sr, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024, cut_silence=False)
        # https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
        self.speaker_embedding_func = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                                     run_opts={"device": str(device)},
                                                                     savedir="Models/SpeakerEmbedding/speechbrain_speaker_embedding_ecapa")

    def extract_condition_from_reference_wave(self, wave, already_normalized=False):
        if already_normalized:
            norm_wave = wave
        else:
            norm_wave = self.ap.audio_to_wave_tensor(normalize=True, audio=wave)
            norm_wave = torch.tensor(trim_zeros(norm_wave.numpy()))
        spk_emb = self.speaker_embedding_func.encode_batch(wavs=norm_wave.unsqueeze(0)).squeeze()
        combined_utt_condition = spk_emb.cpu()
        return combined_utt_condition


if __name__ == '__main__':
    wave, sr = sf.read("../audios/1.wav")
    ext = ProsodicConditionExtractor(sr=sr)
    print(ext.extract_condition_from_reference_wave(wave=wave).shape)
