from speechbrain.inference.separation import SepformerSeparation as separator
# from speechbrain.inference.enhancement import SpectralMaskEnhancement
import torch
import torchaudio
import time
import sounddevice as sd
import numpy as np
import soundfile as sf
import os
import argparse

# model = separator.from_hparams(source="speechbrain/sepformer-wham", savedir='pretrained_models/sepformer-wham')

def record_audio(sample_rate):
    print("Recording... (Press Ctrl+C to stop)")
    try:
        # Use a callback to continuously record until interrupted
        myrecording = sd.rec(int(30 * sample_rate), samplerate=sample_rate, channels=2, dtype='float64', blocking=False)
        sd.wait()  # Wait until recording is finished
    except KeyboardInterrupt:
        # Stop recording when interrupted
        print("Recording stopped.")
        return myrecording
    
def play_audio(audio, sample_rate):
    print("Playing...")
    # Check if the audio is mono (shape: [samples,]) and duplicate channels if necessary
    if audio.ndim == 1 or (audio.ndim == 2 and audio.shape[1] == 1):
        audio = np.tile(audio, (2, 1)).T  # Convert mono to stereo by duplicating the mono channel
    elif audio.ndim == 2 and audio.shape[1] > 2:
        # Handle cases where there are more than 2 channels by selecting the first two
        audio = audio[:, :2]
    # Now play the audio
    sd.play(audio, sample_rate)
    sd.wait()  # Wait until playback is finished
    print("Playback finished.")

def main(output_path):
    sample_rate=8000

    audio_file_path = os.path.join(output_path, 'recorded_audio.wav')
    enhanced_audio_path = os.path.join(output_path, 'enhanced_audio.wav')
    print(audio_file_path)

    audio_data = record_audio(sample_rate)
    
    model = separator.from_hparams(source="speechbrain/sepformer-wham-enhancement", 
                                   savedir='pretrained_models/sepformer-wham-enhancement')

    # noisy = torch.tensor(audio_data, dtype=torch.float).unsqueeze(0)
    # lengths = torch.tensor([1.0])
    # enhanced = enhance_model.enhance_batch(noisy, lengths=lengths)
    # enhanced_numpy = enhanced.squeeze(0).cpu().numpy()
    # # enhanced = enhance_model.enhance_batch(audio_data, lengths=torch.tensor([1.]))

    sf.write(audio_file_path, audio_data, sample_rate)

    est_sources = model.separate_file(audio_file_path) 
    # enhanced_audio = est_sources[:, :, 0].detach().cpu().numpy().astype(np.float32)

    torchaudio.save(enhanced_audio_path, est_sources[:, :, 0].detach().cpu(), 8000)

if __name__ == "__main__":
    BASE_PATH = os.getcwd()

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default=BASE_PATH, type=str)
    args = parser.parse_args()

    main(args.output_dir)