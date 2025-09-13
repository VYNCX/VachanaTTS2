from .config import SpeechConfig
from .voice import Voice
import wave


def load_voice(voice_name="TH_F_1"):
    try:
        voice = Voice.load(f"model/{voice_name}.onnx")    
    except:

        from huggingface_hub import hf_hub_download

        model_filename = f"voice/{voice_name}.onnx"
        config_filename = f"voice/{voice_name}.onnx.json"

        model_path = hf_hub_download(repo_id="VIZINZTOR/Vachana-TTS", filename=model_filename,cache_dir="./model")
        config_path = hf_hub_download(repo_id="VIZINZTOR/Vachana-TTS", filename=config_filename,cache_dir="./model")
        voice = Voice.load(model_path, config_path)

    return voice

def generate_speech(
    text,
    voice="TH_F_1",
    output="test.wav",
    volume=1.0,
    speed=1.0,
    noise_scale=0.667,
    noise_w_scale=0.8
):

    syn_config = SpeechConfig(
        volume=volume,
        length_scale=(1 / speed), 
        noise_scale=noise_scale,
        noise_w_scale=noise_w_scale, 
    )

    voice = load_voice(voice)
    with wave.open(output, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file, syn_config)


    return "Speech Generated"
