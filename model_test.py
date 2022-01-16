
from scipy.io.wavfile import write as write_wav
from infer import restore_model, load_audio


#đường dẫn tới checkpoint và file config cho model
config = 'config/quartznet12x1_abcfjwz.yaml'
encoder_checkpoint = 'data/checkpoints/JasperEncoder-STEP-1312684.pt'
decoder_checkpoint = 'data/checkpoints/JasperDecoderForCTC-STEP-1312684.pt'

neural_factory = restore_model(config, encoder_checkpoint, decoder_checkpoint)
print('restore model checkpoint done!')

sig = load_audio("segment.wav")
greedy_hypotheses, beam_hypotheses = neural_factory.infer_signal(sig)
print('greedy predict:{}'.format(greedy_hypotheses))
print('beamLM predict:{}'.format(beam_hypotheses))