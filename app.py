import os
import librosa
from flask import Flask, request, json
# from scipy.io.wavfile import write as write_wav
from loguru import logger

from infer import VietASR

app = Flask(__name__)
app.config["SECRET_KEY"] = "Long@123"
app.config["DEBUG"] = True

config = 'configs/quartznet12x1_vi.yaml'
encoder_checkpoint = 'models/acoustic_model/vietnamese/JasperEncoder-STEP-289936.pt'
decoder_checkpoint = 'models/acoustic_model/vietnamese/JasperDecoderForCTC-STEP-289936.pt'
lm_path = 'models/language_model/3-gram-lm.binary'

vietasr = VietASR(
    config_file=config,
    encoder_checkpoint=encoder_checkpoint,
    decoder_checkpoint=decoder_checkpoint,
    lm_path=lm_path,
    beam_width=50
)

STATIC_DIR = "static"
UPLOAD_DIR = "upload"
RECORD_DIR = "record"

os.makedirs(os.path.join(STATIC_DIR, UPLOAD_DIR), exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, RECORD_DIR), exist_ok=True)

@app.route('/process', methods=['POST'])
def handle_upload():
    _file = request.files['file']
    if _file.filename == '':
        response = app.response_class(
            response=json.dumps({"message": "error"}),
            status=400,
            mimetype='application/json'
        )
        return response
    logger.info(f'file uploaded: {_file.filename}')
    filepath = os.path.join(STATIC_DIR, UPLOAD_DIR, _file.filename)
    _file.save(filepath)
    logger.info(f'saved file to: {filepath}')
    audio_signal, _ = librosa.load(filepath, sr=16000)
    transcript = vietasr.transcribe(audio_signal)
    logger.info(f'transcript: {transcript}')
    response = app.response_class(
        response=json.dumps({
            "transcript": transcript,
            "audiopath": filepath}),
        status=200,
        mimetype='application/json'
    )
    return response


if __name__ == '__main__':
    Flask.run(app, host="0.0.0.0", port=5000, debug=True)