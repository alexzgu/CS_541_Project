from .train import train, test
from .load import convert_songs_to_tensors

# convert_songs_to_tensors(40)

# train(
#     model_name=None,
#     segment_length_ms=20,
#     epochs=50,
#     batch_size=128,
#     lr=0.001,
#     weight_decay=0.001,
# )
print(test('model_20ms_0.6363_test', 20))

