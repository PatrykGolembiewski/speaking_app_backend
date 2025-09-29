import csv
from dataclasses import dataclass, field

from omegaconf import DictConfig

from openspeech.dataclass.configurations import TokenizerConfigs
from openspeech.tokenizers import register_tokenizer
from openspeech.tokenizers.tokenizer import Tokenizer

@dataclass
class LibriSpeechPhonemeTokenizerConfigs(TokenizerConfigs):
    unit: str = field(default="libri_phoneme", metadata={"help": "Unit of vocabulary."})
    vocab_path: str = field(
        default="../../../LibriSpeech/libri_labels.csv", metadata={"help": "Path of phoneme vocabulary file."}
    )


@register_tokenizer("libri_phoneme", dataclass=LibriSpeechPhonemeTokenizerConfigs)
class LibriSpeechPhonemeTokenizer(Tokenizer):
    """
    Tokenizer class in phoneme-units for LibriSpeech.

    Args:
        configs (DictConfig): configuration set.
    """

    def __init__(self, configs: DictConfig):
        super(LibriSpeechPhonemeTokenizer, self).__init__()
        self.vocab_dict, self.id_dict = self.load_vocab(
            vocab_path=configs.tokenizer.vocab_path,
            encoding=configs.tokenizer.encoding,
        )
        self.labels = self.vocab_dict.keys()
        self.sos_id = int(self.vocab_dict[configs.tokenizer.sos_token])
        self.eos_id = int(self.vocab_dict[configs.tokenizer.eos_token])
        self.pad_id = int(self.vocab_dict[configs.tokenizer.pad_token])
        self.blank_id = int(self.vocab_dict[configs.tokenizer.blank_token])
        self.vocab_path = configs.tokenizer.vocab_path
        self.vocab_size: int = field(default=74, metadata={"help": "Size of vocabulary."})

    def __len__(self):
        return len(self.labels)

    def decode(self, labels):
        """
        Converts label to phoneme sequence

        Args:
            labels (numpy.ndarray): number label

        Returns: sentence
            - sentence (str or list): phoneme labels
        """
        if len(labels.shape) == 1:
            sentence = str()
            for label in labels:
                if label.item() == self.eos_id:
                    break
                elif label.item() == self.blank_id:
                    continue
                sentence += self.id_dict[label.item()]
            return sentence

        sentences = list()
        for batch in labels:
            sentence = str()
            for label in batch:
                if label.item() == self.eos_id:
                    break
                elif label.item() == self.blank_id:
                    continue
                sentence += self.id_dict[label.item()]
            sentences.append(sentence)
        return sentences

    def encode(self, sentence):
        """
        Converts phoneme sentence (string) to label ids

        Args:
            sentence (str): space-separated phonemes

        Returns:
            label (str): space-separated label ids
        """
        label = str()

        for ph in sentence:
            try:
                label += str(self.vocab_dict[ch]) + " " 
            except KeyError:
                continue

        return label[:-1]

    def load_vocab(self, vocab_path, encoding="utf-8"):
        """
        Provides phoneme2id, id2phoneme

        Args:
            vocab_path (str): csv file with phoneme labels
            encoding (str): encoding method

        Returns: unit2id, id2unit
        """
        unit2id = dict()
        id2unit = dict()

        try:
            with open(vocab_path, "r", encoding=encoding) as f:
                labels = csv.reader(f, delimiter=",")
                next(labels)

                for row in labels:
                    unit2id[row[1]] = row[0]
                    id2unit[int(row[0])] = row[1]

            return unit2id, id2unit
        except IOError:
            raise IOError("Phoneme label file (csv format) doesn’t exist: {0}".format(vocab_path))
