import medmnist
from medmnist import INFO
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

# 1) Carregar BreastMNIST (28x28, tons de cinza)
data_flag = 'breastmnist'
info = INFO[data_flag]
DataClass = getattr(medmnist, info['python_class'])

train_dataset = DataClass(split='train', download=True)
val_dataset   = DataClass(split='val',   download=True)
test_dataset  = DataClass(split='test',  download=True)

# 2) Extrair arrays e normalizar
X_train = train_dataset.imgs.astype("float32") / 255.0   # (N,28,28)
y_train = train_dataset.labels.astype("float32")         # (N,1)

X_val   = val_dataset.imgs.astype("float32") / 255.0
y_val   = val_dataset.labels.astype("float32")

X_test  = test_dataset.imgs.astype("float32") / 255.0
y_test  = test_dataset.labels.astype("float32")

# 3) Adicionar canal (grayscale) -> (N,28,28,1)
X_train = np.expand_dims(X_train, axis=-1)
X_val   = np.expand_dims(X_val,   axis=-1)
X_test  = np.expand_dims(X_test,  axis=-1)

# 4) Modelo CNN simples para 28x28x1
model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(1e-3),
              loss=BinaryCrossentropy(),
              metrics=['accuracy'])

# 5) TREINO (agora sim)
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=64,
    verbose=1
)

# 6) Avaliação e guardar
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {acc:.4f}")

model.save("model.h5")
print("Modelo guardado em model.h5")
