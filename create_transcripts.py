import os

text_dir = "/mount/arbeitsdaten/textklang/synthesis/Zischler/Primary_Data/Zischler_Hoelderlin_Heidelberg-text.txt"
wav_dir = "/mount/arbeitsdaten/textklang/synthesis/Zischler/Synthesis_Data/Strophen/Zischler_Hoelderlin_Heidelberg"

with open(text_dir, 'r') as text, open(os.path.join(wav_dir, 'transcript.txt'), 'w') as transcript:
        lines = [line for line in text.read().split("\n\n") if line.lstrip()]
        for index, line in enumerate(lines):
            line = line.replace("\n", "~")
            transcript.write(f'segment_{index}\t{line}\n')


# for dir in os.listdir(wav_dir):
#     with open(os.path.join(text_dir, dir + ".txt"), 'r') as text, open(os.path.join(wav_dir, dir, 'transcript.txt'), 'w') as transcript:
#         lines = [line for line in text.read().split("\n") if line.lstrip()]
#         for index, line in enumerate(lines):
#             transcript.write(f'segment_{index}\t{line}\n')
