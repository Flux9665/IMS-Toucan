import soundfile as sf
import torch
import os

from Preprocessing.ArticulatoryCombinedTextFrontend import ArticulatoryCombinedTextFrontend
from Preprocessing.AudioPreprocessor import AudioPreprocessor
from TrainingInterfaces.Text_to_Spectrogram.AutoAligner.Aligner import Aligner
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.DurationCalculator import DurationCalculator

#root = "/projekte/textklang/Primary-Data/Hoelderlin/txt-und-wavs/Zischler"
root = "/mount/arbeitsdaten/textklang/synthesis/Zischler"
poem_name = 'Brod_und_Wein'

audio_path = os.path.join(root, 'Zischler_Hoelderlin_' + poem_name + '.wav')
transcript_path = os.path.join(root, 'Zischler_Hoelderlin_' + poem_name + '-text.txt')
out_dir = os.path.join(root, poem_name)
os.makedirs(out_dir, exist_ok=True)

# loading modules
acoustic_model = Aligner()
acoustic_model.load_state_dict(torch.load("Models/Aligner/aligner.pt", map_location='cpu')["asr_model"])
dc = DurationCalculator(reduction_factor=1)
tf = ArticulatoryCombinedTextFrontend(language="de")

# extract audio
audio, sr = sf.read(audio_path)
ap = AudioPreprocessor(input_sr=sr, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024, cut_silence=True)
norm_wave = ap.audio_to_wave_tensor(normalize=True, audio=audio)
melspec = ap.audio_to_mel_spec_tensor(audio=norm_wave, normalize=False, explicit_sampling_rate=16000).transpose(0, 1)

# extract phonemes
lines = list()
with open(transcript_path, "r", encoding="utf8") as transcript:
    lines_list = [line for line in transcript.read().split("\n") if line.lstrip()]
    for line in lines_list:
        lines.append(tf.string_to_tensor(line, handle_missing=False).squeeze())
# postprocess phonemes: ~ sentence ~ # --> sentence ~ except for the first one, which is ~ sentence ~
processed_lines = list()
for index, line in enumerate(lines):
    if index == 0:
        processed_lines.append(line[:-1])
    else:
        processed_lines.append(line[1:-1])
lines = processed_lines
joined_phonemes = torch.cat(lines, dim=0)

# get durations of each phone in audio
alignment_path = acoustic_model.inference(mel=melspec,
                                          tokens=joined_phonemes,
                                          save_img_for_debug=os.path.join(out_dir, "debug_alignment.png"),
                                          return_ctc=False)
durations = dc(torch.LongTensor(alignment_path), vis=None).squeeze()

# cut audio according to duration sum of each line in transcript
line_lens = [len(x) for x in lines]
index = 0
segment_durations = list()
for num_phones in line_lens:
    segment_durations.append(sum(durations[index: index + num_phones]))
    index += num_phones
spec_to_wave_factor = len(norm_wave) / sum(segment_durations)
wave_segment_lens = [int(x * spec_to_wave_factor) for x in segment_durations]
index = 0
wave_segments = list()
for i, segment_len in enumerate(wave_segment_lens):
    if i == len(wave_segment_lens) - 1:
        wave_segments.append(norm_wave[index:])
    else:
        wave_segments.append(norm_wave[index: index + segment_len])
        index += segment_len

# write the segments into new files
for index, wave_segment in enumerate(wave_segments):
    sf.write(os.path.join(out_dir, f"segment_{index}.wav"), wave_segment.numpy(), 16000)
    #sf.write(f"split/segment_{index}.wav", wave_segment.numpy(), 16000)
