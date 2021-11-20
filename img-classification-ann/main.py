from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

"""" LOAD DATASET """
imagePaths = 'C:\\Users\\LENOVO\\Documents\\KULIAH\\Semester 5\\Kecerdasan Buatan\\Tubes\\Project AI\\Dataset\\'
label_list = ['Belimbing_contrast', 'Seledri_contrast']
data = []
labels = []

for label in label_list:
    for imagePath in glob.glob(imagePaths + label + '\\*.jpg'):
        # print(imagePath)
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (32, 32))
        data.append(image)
        labels.append(label)

# print(np.array(data).shape)


""" DATA PREPROSESSING """
# ubah type data dari list menjadi array
# ubah nilai dari tiap pixel menjadi range [0..1]
data = np.array(data, dtype='float') / 255.0
labels = np.array(labels)

# ubah nilai dari labels menjadi binary
lb = LabelEncoder()
labels = lb.fit_transform(labels)

# print(labels)


""" SPLIT DATASET """
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
# print('Ukuran data train =', x_train.shape)
# print('Ukuran data test =', x_test.shape)


""" BUILD ANN ARCHITECTURE """
model = Sequential()
model.add(Flatten(input_shape=(32, 32, 3)))
model.add(Dense(1024, activation="relu"))
model.add(Dense(1024, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# print(model.summary())

# tentukan hyperparameter
lr = 0.01
max_epochs = 100
opt_funct = SGD(learning_rate=lr)

# compile arsitektur yang telah dibuat
model.compile(loss='binary_crossentropy', optimizer=opt_funct, metrics=['accuracy'])


""" TRAIN MODEL """
H = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=max_epochs, batch_size=32)
N = np.arange(0, max_epochs)

plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
# plt.plot(N, H.history["accuracy"], label="train_acc")
# plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.xlabel("Epoch #")
plt.legend()
plt.show()


""" EVALUATE THE MODEL """
# menghitung nilai akurasi model terhadap data test
predictions = model.predict(x_test, batch_size=32)
target = (predictions > 0.5).astype(np.int)
print(classification_report(y_test, target, target_names=label_list))

# uji model menggunakan image lain
queryPath = imagePaths + 'query_belimbing.jpg'
query = cv2.imread(queryPath)
output = query.copy()
query = cv2.resize(query, (32, 32))

q = [query]
q = np.array(q, dtype='float') / 255.0

q_pred = model.predict(q)
print(q_pred)

if q_pred <= 0.5:
    target = "Belimbing"
else:
    target = "Seledri"

text = "{}".format(target)
cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# menampilkan output image
cv2.imshow('Output', output)
cv2.waitKey()  # image tidak akan diclose,sebelum user menekan sembarang tombol
cv2.destroyWindow('Output')  # image akan diclose"
