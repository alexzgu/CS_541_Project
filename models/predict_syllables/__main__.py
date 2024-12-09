from .train import train, test
from .syllables import syllables
from .load import convert_songs_to_tensors
from .predict import predict

# convert_songs_to_tensors(40)

# train(
#     model_name=None,
#     segment_length_ms=20,
#     epochs=50,
#     batch_size=128,
#     lr=0.001,
#     weight_decay=0.001,
# )


# accuracy, y, y_hat = test(model_name='model_20ms_drop_0.5_144_0.5363_test', segment_length_ms=20)

predict(model_name='model_20ms_drop_0.5_144_0.5363_test')


