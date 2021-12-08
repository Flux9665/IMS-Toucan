import os
import sys

import torch
import soundfile as sf
from numpy import trim_zeros

from InferenceInterfaces.Eva_FastSpeech2 import Eva_FastSpeech2
from InferenceInterfaces.Karlsson_FastSpeech2 import Karlsson_FastSpeech2
from InferenceInterfaces.Nancy_FastSpeech2 import Nancy_FastSpeech2

from TrainingInterfaces.Text_to_Spectrogram.AutoAligner.Aligner import Aligner
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.DurationCalculator import DurationCalculator
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.EnergyCalculator import EnergyCalculator
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.PitchCalculator import Dio

from run_text_to_file_reader import read_texts
from Utility.path_to_transcript_dicts import build_path_to_transcript_dict_sommer as build_path_to_transcript_dict
#from Utility.path_to_transcript_dicts import build_path_to_transcript_dict_zischler as build_path_to_transcript_dict
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeechDataset import FastSpeechDataset
from Preprocessing.ArticulatoryCombinedTextFrontend import ArticulatoryCombinedTextFrontend
from Preprocessing.AudioPreprocessor import AudioPreprocessor


tts_dict = {
    "fast_nancy"   : Nancy_FastSpeech2,
    "fast_eva"     : Eva_FastSpeech2,
    "fast_karlsson": Karlsson_FastSpeech2,
    }

def extract_prosody(transcript, ref_audio):
    device = 'cpu'
    acoustic_model = Aligner()
    # acoustic_checkpoint_path = os.path.join("Models", "FastSpeech2_Zischler", "aligner", "aligner.pt")
    acoustic_checkpoint_path = os.path.join("Models", "Aligner", "aligner.pt")
    acoustic_model.load_state_dict(torch.load(acoustic_checkpoint_path, map_location='cpu')["asr_model"])

    acoustic_model = acoustic_model.to(device)
    dio = Dio(reduction_factor=1, fs=16000)
    energy_calc = EnergyCalculator(reduction_factor=1, fs=16000)
    dc = DurationCalculator(reduction_factor=1)

    wave, sr = sf.read(ref_audio)
    dur_in_seconds = len(wave) / sr

    tf = ArticulatoryCombinedTextFrontend(language='de', use_word_boundaries=False)
    ap = AudioPreprocessor(input_sr=sr, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024)
    
    try:
        norm_wave = ap.audio_to_wave_tensor(normalize=True, audio=wave)
    except ValueError:
            print('Error')
    dur_in_seconds = len(norm_wave) / 16000
    norm_wave = torch.tensor(trim_zeros(norm_wave.numpy()))
    norm_wave_length = torch.LongTensor([len(norm_wave)])
    # raw audio preprocessing is done

    try:
        text = tf.string_to_tensor(transcript, handle_missing=False).squeeze(0)
    except KeyError:
        #tf.string_to_tensor(transcript, handle_missing=True).squeeze(0).cpu().numpy()
        print("we skip sentences with unknown symbols")
    try:
        if len(text[0]) != 66:
            print(f"There seems to be a problem with the following transcription: {transcript}")

    except TypeError:
            print(f"There seems to be a problem with the following transcription: {transcript}")

    cached_text_len = torch.LongTensor([len(text)]).numpy()
    melspec = ap.audio_to_mel_spec_tensor(audio=norm_wave, normalize=False, explicit_sampling_rate=16000).transpose(0, 1)
    melspec_length = torch.LongTensor([len(melspec)]).numpy()


    alignment_path = acoustic_model.inference(mel=melspec.to(device),
                                            tokens=text.to(device),
                                            return_ctc=False)

    duration = dc(torch.LongTensor(alignment_path), vis=None).cpu()

    energy = energy_calc(input_waves=norm_wave.unsqueeze(0),
                                input_waves_lengths=norm_wave_length,
                                feats_lengths=melspec_length,
                                durations=duration.unsqueeze(0),
                                durations_lengths=torch.LongTensor([len(duration)]))[0].squeeze(0).cpu()
    pitch = dio(input_waves=norm_wave.unsqueeze(0),
                        input_waves_lengths=norm_wave_length,
                        feats_lengths=melspec_length,
                        durations=duration.unsqueeze(0),
                        durations_lengths=torch.LongTensor([len(duration)]))[0].squeeze(0).cpu()
    phones = tf.get_phone_string(transcript)
    print(len(phones), " ", len(duration), " ", len(pitch), " ", len(energy))
    return transcript, duration, pitch, energy

def sort_dataset(text_list, dataset):
    tf = ArticulatoryCombinedTextFrontend(language='de', use_word_boundaries=False)
    sorted_dataset = []
    for text in text_list:
        text_vector = tf.string_to_tensor(text, handle_missing=False).squeeze(0).cpu()
        # print(len(text_vector))
        found_datapoint = False
        for datapoint in dataset:
            if torch.equal(datapoint[0],text_vector):
                sorted_dataset.append(datapoint)
                found_datapoint = True
                # print('found sentence ', str(i))
                break
        if not found_datapoint:
            print("WARNING: No matching datapoint found for sentence ", text)
    return sorted_dataset
        


if __name__ == '__main__':

    exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.isdir("audios"):
        os.makedirs("audios")

    root = "/mount/arbeitsdaten/textklang/synthesis/Zischler/Synthesis_Data/"
    poem = "Der_Sommer"
    transcript = "Der Sommer"
    ref = os.path.join(root, poem, 'segment_0.wav')
    text, duration, pitch, energy = extract_prosody(transcript, ref)

    # text = [text]
    # durations = [duration]
    # pitch = [pitch]
    # energy = [energy]

    

    # cache_dir = "/mount/arbeitsdaten/dialog-1/kochja/projects/cache/sommer"
    # save_dir = os.path.join("Models", "FastSpeech2_Karlsson")
    # path_to_transcript_dict = build_path_to_transcript_dict()
    
    # acoustic_checkpoint_path = os.path.join("Models", "FastSpeech2_Zischler", "aligner", "aligner.pt")

    # print("Preparing Dataset")
    # dataset = FastSpeechDataset(path_to_transcript_dict,
    #                             cache_dir=cache_dir,
    #                             acoustic_checkpoint_path=acoustic_checkpoint_path,
    #                             loading_processes=1,
    #                             max_len_in_seconds=70,
    #                             min_len_in_seconds=0,
    #                             lang="de",
    #                             device='cpu',
    #                             rebuild_cache=False)
    # sentences = ["""  Der Sommer.
    #             Die Tage gehn vorbei mit sanfter Lüfte Rauschen,
    #             Wenn mit der Wolke sie der Felder Pracht vertauschen,
    #             Des Tales Ende trifft der Berge Dämmerungen,
    #             Dort, wo des Stromes Wellen sich hinabgeschlungen.
    #             Der Wälder Schatten sieht umhergebreitet,
    #             Wo auch der Bach entfernt hinuntergleitet,
    #             Und sichtbar ist der Ferne Bild in Stunden,
    #             Wenn sich der Mensch zu diesem Sinn gefunden.
    #             den vierundzwanzigsten Mai siebzehnhundertachtundfünfzig. Scardanelli.  """]


    # sorted_dataset = sort_dataset(sentences, dataset)

    # _, _, _ , _, durations, energy, pitch = list(zip(*sorted_dataset))


    tts = tts_dict["fast_karlsson"](device='cpu')
    d_outs, p_outs, e_outs = tts.predict_prosody(transcript)
    # d_outs = torch.add(d_outs, 2)
    # p_outs = torch.add(p_outs, 1)
    e_outs = torch.add(e_outs, 1)
    print(d_outs, "\n", p_outs, "\n", e_outs)
    
    read_texts(model_id="fast_karlsson",
               sentence=transcript,
               dur_list=None,
               pitch_list=None,
               energy_list=[e_outs],
               device='cpu',
               filename="audios/sommer_01_pred_en_plus2.wav")
    