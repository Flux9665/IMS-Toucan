from tqdm import tqdm
import os

from Preprocessing.Phone2VecTextFrontend import Phone2VecTextFrontend

    
def phonemize_en():
    root = "/mount/arbeitsdaten/dialog-1/kochja/projects/Articulatory_Toucan/Corpora"
    tf = Phone2VecTextFrontend(language="en", use_word_boundaries=True)
    with open(os.path.join(root, "en_gum-ud-train.conllu"), "r", encoding="utf8") as corpus, open(os.path.join(root, 'en_phones.txt'), 'w', encoding="utf8") as file:
        for line in tqdm(corpus):
            if line.startswith('# text ='):
                text = line.split('=', 1)[1].strip()
                phones = tf.get_phone_string(text)
                file.write(phones + '\n')
    print("Done with GUM Corpus")

def phonemize_de():
    root = "/mount/arbeitsdaten/dialog-1/kochja/projects/Articulatory_Toucan/Corpora"
    tf = Phone2VecTextFrontend(language="de", use_word_boundaries=True)
    with open(os.path.join(root, "de_gsd-ud-train.conllu"), "r", encoding="utf8") as corpus, open(os.path.join(root, 'de_phones.txt'), 'w', encoding="utf8") as file:
        for line in tqdm(corpus):
            if line.startswith('# text ='):
                text = line.split('=', 1)[1].strip()
                phones = tf.get_phone_string(text)
                #file.write(phones + '\n')
                print(phones)
    print("Done with GSD Corpus")

if __name__ == '__main__':
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    phonemize_de()
