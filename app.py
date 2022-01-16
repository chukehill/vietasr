import sys
from flask import Flask, request, jsonify
import traceback

from infer import restore_model, load_audio

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import soundfile as sf
import torch

app = Flask("speech2text")
cache = {}


def load_model():
    # https://github.com/dangvansam98/demo_vietasr
    config = 'config/quartznet12x1_abcfjwz.yaml'
    encoder_checkpoint = 'data/checkpoints/JasperEncoder-STEP-1312684.pt'
    decoder_checkpoint = 'data/checkpoints/JasperDecoderForCTC-STEP-1312684.pt'

    neural_factory = restore_model(config, encoder_checkpoint, decoder_checkpoint)
    
    # https://huggingface.co/nguyenvulebinh/wav2vec2-base-vietnamese-250h
    # load model and tokenizer
    processor = Wav2Vec2Processor.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")
    model = Wav2Vec2ForCTC.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")

    return neural_factory, processor, model


def analyze_from_path(path):
    sig = load_audio(path)
    greedy_hypotheses, beam_hypotheses = cache["neural_factory"].infer_signal(sig)
    
    speech, _ = sf.read(path)
    # tokenize
    input_values = processor(speech, return_tensors="pt", padding="longest").input_values  # Batch size 1

    # retrieve logits
    logits = model(input_values).logits

    # take argmax and decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    return {
        "model1": greedy_hypotheses,
        "model2": beam_hypotheses,
        "model3": transcription[0]
    }
    
@app.route("/", methods=["POST"])
def analyze():
    try:
        json_ = request.json
        
        text_result = {}
        if json_ is not None:
            if "path" in json_:
                text_result = analyze_from_path(json_["path"])

        return jsonify(text_result)

    except:

        return jsonify({"trace": traceback.format_exc()})


if __name__ == "__main__":
    try:
        ip = int(sys.argv[1]) 
    except:
        ip = "127.0.0.1"

    print("Restoring model.")
    neural_factory, processor, model = load_model()
    cache["neural_factory"] = neural_factory
    cache["processor"] = processor
    cache["model"] = model
    print("Model restored.")

    app.run(host=ip, port=3000, debug=True) # , debug=True