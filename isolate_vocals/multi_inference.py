import os
from audio_separator.separator import Separator

input = "content/Imperial Circus Dead Decadence - Uta.mp3"
output = "content/output"

separator = Separator(output_dir=output)

#vocals
vocals = os.path.join(output, 'Vocals.wav')
instrumental = os.path.join(output, 'Instrumental.wav')

#vocals w/ reverb
vocals_reverb = os.path.join(output, 'Vocals (Reverb).wav')
vocals_no_reverb = os.path.join(output, 'Vocals (No Reverb).wav')

#lead vocals
lead_vocals = os.path.join(output, 'Lead Vocals.wav')
backing_vocals = os.path.join(output, 'Backing Vocals.wav')


separator.load_model(model_filename='model_bs_roformer_ep_317_sdr_12.9755.ckpt')
voc_inst = separator.separate(input)
os.rename(os.path.join(output, voc_inst[0]), instrumental)
os.rename(os.path.join(output, voc_inst[1]), vocals)

separator.load_model(model_filename='UVR-DeEcho-DeReverb.pth')
voc_no_reverb = separator.separate(vocals)
os.rename(os.path.join(output, voc_no_reverb[0]), vocals_no_reverb)
os.rename(os.path.join(output, voc_no_reverb[1]), vocals_reverb)

separator.load_model(model_filename='mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt')
backing_voc = separator.separate(vocals_no_reverb)
os.rename(os.path.join(output, backing_voc[0]), backing_vocals) .rename(os.path.join(output, backing_voc[1]), lead_vocals) 