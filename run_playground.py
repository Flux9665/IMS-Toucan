import os
import sys

import torch
import soundfile as sf
from numpy import trim_zeros

from InferenceInterfaces.Karlsson_FastSpeech2 import Karlsson_FastSpeech2

from TrainingInterfaces.Text_to_Spectrogram.AutoAligner.Aligner import Aligner
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.DurationCalculator import DurationCalculator
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.EnergyCalculator import EnergyCalculator
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.PitchCalculator import Dio

from run_text_to_file_reader import read_texts
from Preprocessing.ArticulatoryCombinedTextFrontend import ArticulatoryCombinedTextFrontend
from Preprocessing.AudioPreprocessor import AudioPreprocessor


tts_dict = {
    # "fast_nancy"   : Nancy_FastSpeech2,
    # "fast_eva"     : Eva_FastSpeech2,
    "fast_karlsson": Karlsson_FastSpeech2,
    }

#exec_device = "cuda" if torch.cuda.is_available() else "cpu"
exec_device = 'cpu'

def extract_prosody_from_poem(poem):

    root = "/mount/arbeitsdaten/textklang/synthesis/Zischler/Synthesis_Data/"

    device = exec_device
    acoustic_model = Aligner()
    acoustic_checkpoint_path = os.path.join("Models", "FastSpeech2_Zischler", "aligner", "aligner.pt")
    #acoustic_checkpoint_path = os.path.join("Models", "Aligner", "aligner.pt")
    acoustic_model.load_state_dict(torch.load(acoustic_checkpoint_path, map_location='cpu')["asr_model"])

    acoustic_model = acoustic_model.to(device)
    dio = Dio(reduction_factor=1, fs=16000)
    energy_calc = EnergyCalculator(reduction_factor=1, fs=16000)
    dc = DurationCalculator(reduction_factor=1)
    
    text_list = []
    dur_list = []
    pitch_list = []
    en_list = []

    with open(os.path.join(root, poem, "transcript.txt")) as transcript:
        text_list = transcript.read().split("\n")
        text_list = [text.split("\t")[1] for text in text_list]

    for index, text in enumerate(text_list):
        ref = os.path.join(root, poem, f'segment_{index}.wav')
        # print(ref, "\t", text)
        duration, pitch, energy = extract_prosody(text, ref, acoustic_model, dio, energy_calc, dc)
        dur_list.append(duration)
        pitch_list.append(pitch)
        en_list.append(energy)
    
    return text_list, dur_list, pitch_list, en_list

    

def extract_prosody(transcript, ref_audio, acoustic_model, dio, energy_calc, dc):
    
    wave, sr = sf.read(ref_audio)
    device = exec_device

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

    melspec = ap.audio_to_mel_spec_tensor(audio=norm_wave, normalize=False, explicit_sampling_rate=16000).transpose(0, 1)
    melspec_length = torch.LongTensor([len(melspec)]).numpy()


    alignment_path = acoustic_model.inference(mel=melspec.to(device),
                                            tokens=text.to(device),
                                            return_ctc=False)

    duration = dc(torch.LongTensor(alignment_path), vis=None)

    energy = energy_calc(input_waves=norm_wave.unsqueeze(0),
                                input_waves_lengths=norm_wave_length,
                                feats_lengths=melspec_length,
                                durations=duration.unsqueeze(0),
                                durations_lengths=torch.LongTensor([len(duration)]))[0].squeeze(0)
    pitch = dio(input_waves=norm_wave.unsqueeze(0),
                        input_waves_lengths=norm_wave_length,
                        feats_lengths=melspec_length,
                        durations=duration.unsqueeze(0),
                        durations_lengths=torch.LongTensor([len(duration)]))[0].squeeze(0)

    # print("len melspec: ", melspec_length, "\tduration: ", sum(duration))
    # phones = tf.get_phone_string(transcript)
    # print(phones)
    # print(len(phones), " ", len(duration), " ", len(pitch), " ", len(energy))
    return duration, pitch, energy
        


if __name__ == '__main__':

    if not os.path.isdir("audios"):
        os.makedirs("audios")

    #exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    acoustic_model = Aligner()
    acoustic_checkpoint_path = os.path.join("Models", "FastSpeech2_Zischler", "aligner", "aligner.pt")
    acoustic_model.load_state_dict(torch.load(acoustic_checkpoint_path, map_location='cpu')["asr_model"])


    acoustic_model = acoustic_model.to('cpu')
    dio = Dio(reduction_factor=1, fs=16000)
    energy_calc = EnergyCalculator(reduction_factor=1, fs=16000)
    dc = DurationCalculator(reduction_factor=1)

    text = "Die Tage gehn vorbei mit sanfter Lüfte Rauschen,"
    d_gold, p_gold, e_gold = extract_prosody(text, "/mount/arbeitsdaten/textklang/synthesis/Zischler/Synthesis_Data/Der_Sommer/segment_1.wav", acoustic_model, dio, energy_calc, dc)

    tts = tts_dict["fast_karlsson"](device='cpu')
    phones, d_pred, p_pred, e_pred = tts.predict_prosody(text)

    print("d_gold: ", p_gold.shape, "\td_pred: ", p_pred.shape)
    print('d_pred type: ', type(d_pred))
    print(d_pred)
    print(d_gold)

    w_gold = 0.8
    w_pred = 1 - w_gold

    d_gold = d_gold * w_gold
    p_gold = p_gold * w_gold
    e_gold = e_gold * w_gold

    d_pred = d_pred * w_pred
    p_pred = p_pred * w_pred
    e_pred = e_pred * w_pred
    
    d_out = torch.add(d_gold, d_pred).long()
    p_out = torch.add(p_gold, p_pred)
    e_out = torch.add(e_gold, e_pred)

   
    print('d_out type: ', type(d_out))
    print(d_out)
    # print(torch.equal(d_out, d_pred))
    read_texts(model_id="fast_karlsson",
               sentence=[text],
               dur_list=[d_out],
               pitch_list=[p_out],
               energy_list=[e_out],
               device='cpu',
               filename=f"audios/Test/test_Sommer_Jambus.wav")

    sys.exit(0)
    text = "Einst stritten sich Nordwind und Sonne"
    tts = tts_dict["fast_karlsson"](device='cpu')
    phones, d_outs, p_outs, e_outs = tts.predict_prosody(text)
    
    print()
    # d_outs = torch.add(d_outs, 2)
    # p_outs = torch.add(p_outs, 1)
    # e_outs = torch.add(e_outs, 1)
    print(phones, "\n", d_outs) #, "\n", p_outs, "\n", e_outs)
    print(type(d_outs))
    read_texts(model_id="fast_karlsson",
               sentence=[text],
               dur_list=[d_outs],
               pitch_list=[p_outs],
               energy_list=[e_outs],
               device='cpu',
               filename=f"audios/test.wav")
    
    sys.exit(0)
    poem = "Nachtgesaenge_09_Der_Winkel_von_Hardt"
    text_list, dur_list, pitch_list, en_list = extract_prosody_from_poem(poem)
    i = 0
    if not os.path.isdir(f"audios/{poem}"):
        os.makedirs(f"audios/{poem}")
    for text, dur, pitch, en in zip(text_list, dur_list, pitch_list, en_list):

        read_texts(model_id="fast_karlsson",
                    sentence=[text],
                    dur_list=[dur],
                    pitch_list=[pitch],
                    energy_list=[en],
                    device=exec_device,
                    filename=f"audios/{poem}/segment_{i}.wav")
        i += 1

    print()    
    read_texts(model_id="fast_karlsson",
               sentence=text_list,
               dur_list=dur_list,
               pitch_list=pitch_list,
               energy_list=en_list,
               device=exec_device,
               filename=f"audios/{poem}/{poem}.wav")
    