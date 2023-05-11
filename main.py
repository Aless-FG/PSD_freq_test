import sklearn
from matplotlib import pyplot
from sklearn.decomposition import PCA
from scipy import signal
import numpy as np
import pandas as pd
from scipy.io import loadmat
import pywt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import f1_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from scipy.stats import kurtosis, skew


def calculate_spectral_crest_factor(psd):
    crest_factor = np.max(psd, axis=0) / np.mean(psd, axis=0)
    return crest_factor


def calculate_spectral_flatness(psd):
    geometric_mean = np.exp(np.mean(np.log(psd)))
    arithmetic_mean = np.mean(psd, axis=0)
    flatness = geometric_mean / arithmetic_mean
    return flatness



n_subjects = 21
n_sessions = 3
data_array = np.zeros((21, 3, 14, 370, 11))  # soggetti, sessioni, canali, psd, bande di frequenza
for i in range(1, 22):
    for j in range(1, 4):

        s_mat = loadmat(f"/home/ale/Desktop/BED_Biometric_EEG_dataset/BED/RAW_PARSED/s{i}_s{j}.mat")
        rec = s_mat['recording']
        rec_dataframe = pd.DataFrame(rec,
                                     columns=['COUNTER', 'INTERPOLATED', 'F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1',
                                              'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4', 'UNIX_TIMESTAMP'])
        sensors = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4'] # channels
        z = 0
        matrix_feature = np.zeros((370, 11))  # perchè 370 sono il numero di psd, 5 sono le features
        for sensor in sensors:
            sig = rec_dataframe[sensor]
            fs = 256  # frequenza di campionamento
            fc_hp = 0.5  # Frequenza di taglio inferiorerecall, sensitivity, precision = calculate_metrics(y_true, y_pred)
            fc_lp = 45  # Frequenza di taglio superiore
            N = 101
            order = 4  # Ordine del filtro
            h_lp = signal.firwin(N, fc_lp / (fs / 2), window='flattop')
            h_hp = signal.firwin(N, fc_hp / (fs / 2), window='flattop', pass_zero=False)
            # Applicazione del filtro passa basso al segnale x
            x_lp = signal.filtfilt(h_lp, 1, sig)
            # Applicazione del filtro passa alto al segnale x
            x_hp = signal.filtfilt(h_hp, 1, sig)
            x_lp = signal.filtfilt(h_lp, 1, sig)
            # Applicazione del filtro passa alto al segnale x
            x_hp = signal.filtfilt(h_hp, 1, x_lp)
            wavelet = pywt.Wavelet('db4')
            coeffs = pywt.wavedec(x_hp, wavelet, level=2)
            # Esempio di calcolo della TD-PSD su una sub-banda di frequenza utilizzando la finestra di Hamming
            f, t_psd, psd = signal.spectrogram(coeffs[0], fs, window='blackman', nperseg=fs, noverlap=int(fs / 4)
                                               )
            psd_norm = sklearn.preprocessing.normalize(psd, axis=0)  # normalization, axis = 0 sono le frequenze
            if len(psd_norm[0]) > 370:
                psd_norm = psd_norm[:, :370]
            elif len(psd_norm[0]) < 370:
                mean = np.mean(psd_norm[len(psd_norm) - 1], axis=0)
                matrix = np.full((129, 370 - len(psd_norm[0])), mean)
                psd_norm = np.concatenate((psd_norm, matrix), axis=1)
            # Calcolo delle frazioni di potenza
            delta_band = (1, 4)
            theta_band = (4, 8)
            alpha_band = (8, 12)
            beta_band = (12, 30)
            gamma_band = (30, 100)
            # --- features ---
            delta_power = np.sum(psd_norm[np.where(np.logical_and(f >= delta_band[0], f <= delta_band[1]))],
                                 axis=0)  # somma delle frequenze delta (tra 1 e 4, è la potenza)
            matrix_feature[:, 0] = delta_power  # aggiungo alla prima colonna
            theta_power = np.sum(psd_norm[np.where(np.logical_and(f >= theta_band[0], f <= theta_band[1]))], axis=0)
            matrix_feature[:, 1] = theta_power
            alpha_power = np.sum(psd_norm[np.where(np.logical_and(f >= alpha_band[0], f <= alpha_band[1]))], axis=0)
            matrix_feature[:, 2] = alpha_power
            beta_power = np.sum(psd_norm[np.where(np.logical_and(f >= beta_band[0], f <= beta_band[1]))], axis=0)
            matrix_feature[:, 3] = beta_power
            gamma_power = np.sum(psd_norm[np.where(np.logical_and(f >= gamma_band[0], f <= gamma_band[1]))], axis=0)
            matrix_feature[:, 4] = gamma_power
            spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm), axis=0)
            min_value_entropy = np.min(spectral_entropy)
            max_value_entropy = np.max(spectral_entropy)
            normalized_entropy = (spectral_entropy - min_value_entropy) / (max_value_entropy - min_value_entropy)
            matrix_feature[:, 5] = normalized_entropy
            peak_power = np.max(psd_norm, axis=0)
            matrix_feature[:, 6] = peak_power
            dominant_freq = f[np.argmax(psd_norm, axis=0)]
            dominant_freq_norm = (dominant_freq - np.min(dominant_freq)) / (
                    np.max(dominant_freq) - np.min(dominant_freq))
            matrix_feature[:, 7] = dominant_freq_norm
            signal_energy = np.sum(psd_norm, axis=0)
            signal_energy_norm = (signal_energy - np.min(signal_energy)) / (
                    np.max(signal_energy) - np.min(signal_energy))
            matrix_feature[:, 8] = signal_energy_norm
            crest_factor = calculate_spectral_crest_factor(psd_norm)
            crest_factor_norm = (crest_factor - np.min(crest_factor)) / (np.max(crest_factor) - np.min(crest_factor))
            matrix_feature[:, 9] = crest_factor_norm
            flatness = calculate_spectral_flatness(psd_norm)
            flatness_norm = (flatness - np.min(flatness)) / (np.max(flatness) - np.min(flatness))
            matrix_feature[:, 10] = flatness_norm



            data_array[i - 1, j - 1, z, :, :] = matrix_feature[:, :]
            z = z + 1
new_shape = (data_array.shape[0] * data_array.shape[1] * data_array.shape[2], data_array.shape[3], data_array.shape[4])
data_reshape = data_array.reshape(new_shape)
# crea un array numpy di valori interi da 1 a 21
labels = np.repeat(np.arange(0, 21), 42)
# Suddivisione dati in training set e test set
X_train, X_test, y_train, y_test = train_test_split(data_reshape, labels, test_size=0.3, random_state=42, shuffle=True)

# Costruzione modello LSTM
model = Sequential()
model.add(LSTM(64, input_shape=(370, 11), dropout=0.2))
model.add(Dropout(0.20))
model.add(Dense(32, activation='relu'))
model.add(Dense(21, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Addestramento modello
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=150, batch_size=32)

# Valutazione modello su test set
test_loss, test_acc = model.evaluate(X_test, y_test)

y_pred = model.predict(X_test)

# print(y_test)
y_pred = np.argmax(y_pred, axis=1)
# print(y_pred)
f1score = f1_score(y_test, y_pred, average='weighted')
print('F1-score:', f1score)
# precision = precision_score(y_test, y_pred)
print("Recall:", sklearn.metrics.recall_score(y_test, y_pred, average="weighted"))
