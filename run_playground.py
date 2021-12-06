import os

import torch

from InferenceInterfaces.Eva_FastSpeech2 import Eva_FastSpeech2
from InferenceInterfaces.Karlsson_FastSpeech2 import Karlsson_FastSpeech2
from InferenceInterfaces.Nancy_FastSpeech2 import Nancy_FastSpeech2

from run_text_to_file_reader import read_texts
from Utility.path_to_transcript_dicts import build_path_to_transcript_dict_sommer as build_path_to_transcript_dict
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeechDataset import FastSpeechDataset
from Preprocessing.ArticulatoryCombinedTextFrontend import ArticulatoryCombinedTextFrontend

tts_dict = {
    "fast_nancy"   : Nancy_FastSpeech2,
    "fast_eva"     : Eva_FastSpeech2,
    "fast_karlsson": Karlsson_FastSpeech2,
    }

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
            print("WARNING: No matching datapoint found")
    return sorted_dataset
        


if __name__ == '__main__':

    exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.isdir("audios"):
        os.makedirs("audios")

    cache_dir = "/mount/arbeitsdaten/dialog-1/kochja/projects/cache/sommer"
    save_dir = os.path.join("Models", "FastSpeech2_Karlsson")
    path_to_transcript_dict = build_path_to_transcript_dict()

    acoustic_checkpoint_path = os.path.join("Models", "Aligner", "aligner.pt")

    print("Preparing Dataset")
    dataset = FastSpeechDataset(path_to_transcript_dict,
                                cache_dir=cache_dir,
                                acoustic_checkpoint_path=acoustic_checkpoint_path,
                                loading_processes=1,
                                max_len_in_seconds=70,
                                lang="de",
                                device='cpu',
                                rebuild_cache=False)

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

    sentences = ["Die Tage gehn vorbei mit sanfter Lüfte Rauschen,",
                "Wenn mit der Wolke sie der Felder Pracht vertauschen,",
                "Des Tales Ende trifft der Berge Dämmerungen,",
                "Dort, wo des Stromes Wellen sich hinabgeschlungen.",
                "Der Wälder Schatten sieht umhergebreitet,",
                "Wo auch der Bach entfernt hinuntergleitet,",
                "Und sichtbar ist der Ferne Bild in Stunden,",
                "Wenn sich der Mensch zu diesem Sinn gefunden."]

    sorted_dataset = sort_dataset(sentences, dataset)

    _, _, _ , _, durations, energy, pitch = list(zip(*sorted_dataset))


    read_texts(model_id="fast_karlsson",
               sentence=sentences,
               dur_list=None,
               pitch_list=None,
               energy_list=energy,
               device='cpu',
               filename="audios/sommer_energy.wav")
    