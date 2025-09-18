from medmnist import BreastMNIST
from tensorflow.keras.models import load_model
import numpy as np

# 1. Carregar conjunto de teste e modelo treinado
ds = BreastMNIST(split='test', download=True)
X_test = ds.imgs.astype("float32") / 255.0
X_test = np.expand_dims(X_test, axis=-1)
y_true = ds.labels.reshape(-1)

model = load_model("model.h5")

# 2. Prever todas as imagens de teste
pred_probs = model.predict(X_test, batch_size=64, verbose=0).reshape(-1)
pred_labels = (pred_probs > 0.5).astype(int)

# 3. Mostrar índice, verdadeiro, previsto e probabilidade
for i, (true_label, pred_label, prob) in enumerate(zip(y_true, pred_labels, pred_probs)):
    print(f"{i:03d}  real:{true_label}  previsto:{pred_label}  prob:{prob:.2f}")

# 4. Resumo geral
accuracy = (pred_labels == y_true).mean()
print("\nAcurácia global:", round(accuracy*100, 2), "%")
