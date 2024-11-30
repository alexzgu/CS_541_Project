from audio_separator.separator import Separator

separator = Separator()
# Load a machine learning model (if unspecified, defaults to 'model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt')
separator.load_model()
output_files = separator.separate('content/condensed-milk.wav')

print(f"Separation complete! Output file(s): {' '.join(output_files)}")