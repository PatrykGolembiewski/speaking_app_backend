from flask import Flask, request, jsonify, send_file
import subprocess
import os
import uuid
from typing import List
from difflib import SequenceMatcher

app = Flask(__name__)

ckpt_path = "/users/project1/pt01277/app/checkpoints/3_60000.ckpt"
vocab_path = "/users/project1/pt01277/app/openspeech/tokenizers/librispeech/libri_phoneme_labels.csv"
manifest_phoneme = "manifest_phoneme"

cmu_phonemes = [
    'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0', 'AO1', 'AO2',
    'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH', 'EH0', 'EH1', 'EH2',
    'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0',
    'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0', 'OY1', 'OY2',
    'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y',
    'Z', 'ZH'
]
phoneme_set = set(cmu_phonemes)

def split_phonemes_best_match(s: str) -> List[str]:
    memo = {}
    def helper(index):
        if index in memo:
            return memo[index]
        if index == len(s):
            return []
        best = None
        for length in range(4, 0, -1):
            part = s[index:index + length]
            if part in phoneme_set:
                rest = helper(index + length)
                if rest is not None:
                    result = [part] + rest
                    if best is None or len(result) < len(best):
                        best = result
        memo[index] = best
        return best
    result = helper(0)
    if result is None:
        print(f"Failed to break: {s}")
        return []
    return result

def align_with_sequence_matcher(ref: List[str], hyp: List[str]):
    matcher = SequenceMatcher(None, ref, hyp)
    aligned_ref, aligned_hyp, labels = [], [], []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            aligned_ref.extend(ref[i1:i2])
            aligned_hyp.extend(hyp[j1:j2])
            labels.extend(["correct"] * (i2 - i1))
        elif tag == "replace":
            len1, len2 = i2 - i1, j2 - j1
            maxlen = max(len1, len2)
            for k in range(maxlen):
                r = ref[i1 + k] if k < len1 else "-"
                h = hyp[j1 + k] if k < len2 else "-"
                aligned_ref.append(r)
                aligned_hyp.append(h)
                labels.append("substitution")
        elif tag == "insert":
            for h in hyp[j1:j2]:
                aligned_ref.append("-")
                aligned_hyp.append(h)
                labels.append("insertion")
        elif tag == "delete":
            for r in ref[i1:i2]:
                aligned_ref.append(r)
                aligned_hyp.append("-")
                labels.append("deletion")
    return aligned_ref, aligned_hyp, labels


def load_manifest():
    exercises = []
    with open(manifest_phoneme, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            text_id = parts[0]
            audio_path = parts[1]
            transcript_tokens = []
            for token in parts[2:]:
                if token.isdigit():
                    break
                transcript_tokens.append(token)
            transcript = " ".join(transcript_tokens)
            exercises.append({
                "text": text_id,
                "audio_path": audio_path,
                "transcript": transcript
            })
    return exercises

def load_manifest_entry(idx: int):
    exercises = load_manifest()
    if 1 <= idx <= len(exercises):
        ex = exercises[idx - 1]
        return ex["text"], ex["audio_path"], ex["transcript"]
    return None, None, None

@app.route("/exercise/transcript", methods=["GET"])
def get_reference_transcript():
    ex_id = request.args.get("id", default=1, type=int)
    text_id, audio_path, transcript = load_manifest_entry(ex_id)
    if not transcript:
        return jsonify({"error": "Invalid exercise id"}), 400
    return jsonify({
        "exercise_id": ex_id,
        "text": text_id,
        "reference_transcript": transcript
    })

@app.route("/exercise/audio", methods=["GET"])
def get_reference_audio():
    ex_id = request.args.get("id", default=1, type=int)
    _, audio_path, _ = load_manifest_entry(ex_id)
    if not audio_path:
        return jsonify({"error": "Invalid exercise id"}), 400
    if os.path.exists(audio_path):
        return send_file(audio_path, as_attachment=True)
    else:
        return jsonify({"error": "Reference audio not found"}), 404

@app.route("/exercise/count", methods=["GET"])
def get_exercise_count():
    exercises = load_manifest()
    return jsonify({"exercise_count": len(exercises)})

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return "No file part", 400
    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400
    uid = uuid.uuid4().hex
    os.makedirs("./uploads", exist_ok=True)
    os.makedirs("./results", exist_ok=True)
    upload_filename = f"./uploads/upload_{uid}.flac"
    result_filename = f"./results/results_{uid}.txt"
    manifest_path = f"{upload_filename}.csv"
    file.save(upload_filename)
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(f"{upload_filename}|unknown\n")
    subprocess.run([
        "python3", "run_model.py",
        "model=deepspeech2",
        "tokenizer=libri_phoneme",
        f"tokenizer.vocab_path={vocab_path}",
        "audio=mfcc",
        f"eval.checkpoint_path={ckpt_path}",
        f"eval.manifest_file_path={manifest_path}",
        f"eval.dataset_path={upload_filename}",
        "eval.use_cuda=false",
        f"eval.result_path={result_filename}",
        "hydra.run.dir=."
    ])

    if os.path.exists(result_filename):
        with open(result_filename, "r", encoding="utf-8") as f:
            transcription = f.read().strip()
    else:
        transcription = "No results found"
    hyp_phonemes = split_phonemes_best_match(transcription)
    ex_id = request.args.get("id", default=1, type=int)
    _, _, reference_transcript = load_manifest_entry(ex_id)
    ref_phonemes = reference_transcript.split() if reference_transcript else []
    aligned_ref, aligned_hyp, labels = align_with_sequence_matcher(ref_phonemes, hyp_phonemes)

    return jsonify({
        "raw_transcription": transcription,
        "split_phonemes": hyp_phonemes,
        "reference_phonemes": ref_phonemes,
        "aligned_ref": aligned_ref,
        "aligned_hyp": aligned_hyp,
        "labels": labels
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

