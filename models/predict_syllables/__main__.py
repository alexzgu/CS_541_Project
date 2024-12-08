from .train import train, test
from .syllables import syllables
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

from matplotlib import rcParams
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


accuracy, y, y_hat = test(model_name='model_20ms_drop_0.5_144_0.5363_test', segment_length_ms=20)

# Set up the Japanese font
rcParams['font.family'] = 'sudo apt install fonts-noto-cjk'

print(accuracy)

# Compute confusion matrix
cm = confusion_matrix(y, y_hat, labels=list(range(110)))

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(45, 45))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=syllables)
disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical', ax=plt.gca())

plt.title("Confusion Matrix", fontsize=25)
ax.tick_params(axis="x", labelsize=16)
ax.tick_params(axis="y", labelsize=16)
plt.show()

