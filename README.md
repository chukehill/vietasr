# VietASR (NVIDIA NeMo ToolKit) - fork
⚡ Some experiment with [NeMo](https://github.com/NVIDIA/NeMo) ⚡
# Result
* Model: [QuartzNet](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/models.html#quartznet) is a smaller version of [Jaser](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/models.html#jasper)  model  
* I list the word error rate (WER) with and without LM of major ASR tasks.

| Task                   | CER (%) | WER (%) | +LM WER (%) |
| -----------            | :----:  | :----:  | :----:                                                                                                                                                                |
| VIVOS (TEST)            |  6.80 | 18.02 | 15.72 |
| VLSP2018                |  6.87 | 16.26 |  N/A  |
| VLSP2020 T1             | 14.73 | 30.96 |  N/A  |
| VLSP2020 T2             | 41.67 | 69.15 |  N/A  |

Model was trained with ~500 hours Vietnamese speech dataset, was collected from youtube, radio, call center(8k), text to speech data and some public dataset (vlsp, vivos, fpt). It is very small model (13M parameters) make it inference so fast ⚡  

# Installation 
## Docker - testing
Build docker image, might take up to 25 mins.

`docker-compose -f docker-compose.yml up --build`

Run docker image.

`docker run -it speech2text /bin/bash`

Example of test.
```
root@764e6366dc01:/home/root/speech2text/demo_vietasr# python model_test.py 
################################################################################
### WARNING, path does not exist: KALDI_ROOT=/mnt/matylda5/iveselyk/Tools/kaldi-trunk
###          (please add 'export KALDI_ROOT=<your_path>' in your $HOME/.profile)
###          (or run as: KALDI_ROOT=<your_path> python <your_script>.py)
################################################################################

/home/root/speech2text/demo_vietasr/nemo/collections/asr/audio_preprocessing.py:48: UserWarning: Could not import torchaudio. Some features might not work.
  warnings.warn('Could not import torchaudio. Some features might not work.')
[NeMo I 2022-01-16 15:30:49 features:149] PADDING: 0
[NeMo I 2022-01-16 15:30:49 features:170] STFT using torch
restore model checkpoint done!
/usr/local/lib/python3.8/dist-packages/torch/functional.py:572: UserWarning: stft will soon require the return_complex parameter be given for real inputs, and will further require that return_complex=True in a future PyTorch release. (Triggered internally at  ../aten/src/ATen/native/SpectralOps.cpp:659.)
  return _VF.stft(input, n_fft, hop_length, win_length, window,  # type: ignore[attr-defined]
greedy predict:mấy tát đểng chí và các bạn nghe chương trình phát anh quân đổi nhân dân
beamLM predict:mấy cái đồng chí và các bạn nghe chương trình phát thanh quân đội nhân dân
```
## Docker - app
```
docker-compose -f docker-compose.yml up --build
docker-compose -f docker-compose_app.yml up --build
```

## App
I included in the `app.py` also the model from [nguyenvulebinh/wav2vec2-base-vietnamese-250h](https://huggingface.co/nguyenvulebinh/wav2vec2-base-vietnamese-250h). For installation of all needed packages run

`python -m pip install nemo_toolkit[all]==1.5.1 flask transformers==4.9.2 soundfile datasets==1.11.0 pyctcdecode==v0.1.0`

Build KenLM library
```shell
cd nemo/scripts/decoders/kenlm
mkdir build
cmake ..
make -j
cd ..
cp build/lib.linux-x86_64-3.8/_swig_decoders.cpython-38-x86_64-linux-gnu.so .
```
Install ctc_decoders

`python3.8 -m pip install ./`

Then test that ctc_decoders are installed

`python ctc_decoders_test.py`

Run the app

`python app.py`
