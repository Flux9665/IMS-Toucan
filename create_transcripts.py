import os

text_dir = "/mount/arbeitsdaten/textklang/synthesis/Maerchen/Wunderhorn-Texte"
wav_dir = "/mount/arbeitsdaten/textklang/synthesis/Maerchen/Synthesis_Data"

for dir in os.listdir(wav_dir):
    with open(os.path.join(text_dir, dir + ".txt"), 'r') as text, open(os.path.join(wav_dir, dir, 'transcript.txt'), 'w') as transcript:
        lines = [line for line in text.read().split("\n") if line.lstrip()]
        for index, line in enumerate(lines):
            transcript.write(f'segment_{index}\t{line}\n')
