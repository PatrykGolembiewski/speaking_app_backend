import logging
import pandas as pd
from nltk.corpus import cmudict
import pronouncing
from openspeech.datasets.librispeech.preprocess.preprocess import collect_transcripts

logger = logging.getLogger(__name__)
cmu = cmudict.dict()

def _generate_phoneme_labels(labels_dest):
    logger.info("Creating phoneme vocab...")

    all_phonemes = set()

    for word, phoneme_lists in cmu.items():
        for phoneme_seq in phoneme_lists:
            all_phonemes.update(phoneme_seq)

    special_tokens = ["<pad>", "<sos>", "<eos>", "<blank>"]
    tokens = special_tokens + sorted(list(all_phonemes))

    label = {
        "id": [x for x in range(len(tokens))],
        "char": tokens,
    }

    label_df = pd.DataFrame(label)
    label_df.to_csv(labels_dest, encoding="utf-8", index=False)


def _load_label(filepath):
    char2id, id2char = dict(), dict()
    ch_labels = pd.read_csv(filepath, encoding="utf-8")

    for (id_, char) in zip(ch_labels["id"], ch_labels["char"]):
        char2id[char] = id_
        id2char[id_] = char
    return char2id, id2char

def get_phonemes(word):
    word = word.lower()
    if word in cmu:
        return cmu[word][0]
    return []

def sentence_to_phoneme_ids(sentence, char2id):
    phoneme_ids = []
    phoneme_symbols = []
    for word in sentence.strip().split():
        phones = pronouncing.phones_for_word(word.lower())
        if phones:
            phonemes = phones[0].split()
            for ph in phonemes:
                if ph in char2id:
                    phoneme_ids.append(str(char2id[ph]))
                    phoneme_symbols.append(ph)
    return " ".join(phoneme_symbols), " ".join(phoneme_ids)


def generate_manifest_files(dataset_path: str, manifest_file_path: str, vocab_path: str) -> None:
    _generate_phoneme_labels(vocab_path)
    char2id, id2char = _load_label(vocab_path)
    transcripts_collection = collect_transcripts(dataset_path)

    with open(manifest_file_path, "w") as f:
        #for idx, part in enumerate(["train-960", "dev-clean", "dev-other", "test-clean", "test-other"]):
        for idx, part in enumerate(["test-clean"]):
            for transcript in transcripts_collection[idx]:
                audio_path, sentence = transcript.split("|")
                phoneme_str, phoneme_ids = sentence_to_phoneme_ids(sentence, char2id)
                if phoneme_ids.strip() == "":
                    continue
                f.write(f"{audio_path}\t{phoneme_str}\t{phoneme_ids}\n")
    return

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--manifest_file_path", type=str, required=True)
    parser.add_argument("--vocab_path", type=str, required=True)
    args = parser.parse_args()

    generate_manifest_files(
        dataset_path=args.dataset_path,
        manifest_file_path=args.manifest_file_path,
        vocab_path=args.vocab_path,
    )
