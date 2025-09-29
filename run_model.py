import logging
import os
import warnings
import numpy as np
import librosa
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_info
from openspeech.dataclass.initialize import hydra_eval_init
from openspeech.models import MODEL_REGISTRY
from openspeech.tokenizers import TOKENIZER_REGISTRY
from openspeech.data import AUDIO_FEATURE_TRANSFORM_REGISTRY

logger = logging.getLogger(__name__)


@hydra.main(config_path=os.path.join("openspeech", "configs"), config_name="eval")
def hydra_main(configs: DictConfig) -> None:
    rank_zero_info(OmegaConf.to_yaml(configs))
    results = []

    audio_path = configs.eval.dataset_path
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"No audio file found: {audio_path}")

    use_cuda = configs.eval.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    tokenizer = TOKENIZER_REGISTRY[configs.tokenizer.unit](configs)
    model_cls = MODEL_REGISTRY[configs.model.model_name]
    model = model_cls.load_from_checkpoint(
        configs.eval.checkpoint_path,
        configs=configs,
        tokenizer=tokenizer
    )
    model.to(device)
    model.eval()

    if configs.eval.beam_size > 1:
        model.set_beam_decoder(beam_size=configs.eval.beam_size)

    target_sr = getattr(configs.audio, "sample_rate", None) or 16000
    signal,_ = librosa.load(audio_path,sr=target_sr)

    mfcc_transform = AUDIO_FEATURE_TRANSFORM_REGISTRY[configs.audio.name](configs)

    mfcc_features = mfcc_transform(signal)
    mfcc_features -= mfcc_features.mean()
    mfcc_features /= np.std(mfcc_features)

    mfcc_features = torch.FloatTensor(mfcc_features).transpose(0, 1)
    if mfcc_features.shape[0] == configs.audio.num_mels:
        mfcc_features = mfcc_features.T
    features_tensor = torch.tensor(mfcc_features, dtype=torch.float32).unsqueeze(0).to(device)
    input_lengths = torch.tensor([features_tensor.size(1)], dtype=torch.int32).to(device)

    logger.info("Start inference ...")
    with torch.no_grad():
        outputs = model(features_tensor, input_lengths)

    if outputs.get('predictions') is not None:
        decoded_preds = tokenizer.decode(outputs['predictions'])
        print("decoded_preds:", decoded_preds)
    else:
        print("No prediction for decoding")
        decoded_preds = []

    for prediction in decoded_preds:
        results.append(prediction + "\n")

    with open(configs.eval.result_path, "wt", encoding="utf-8") as f:
        for result in results:
            f.write(result)

    logger.info(f"The result was saved to: {configs.eval.result_path}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    hydra_eval_init()
    hydra_main()

