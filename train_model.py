## “İşe Alımda Aday Seçimi: SVM ile Başvuru Değerlendirme”

import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import joblib

def train_and_save_svm_model_linear():
    np.random.seed(42)
    n_samples = 200
    years_of_experience = np.random.randint(0, 11, n_samples)
    technical_point = np.random.randint(0, 101, n_samples)
    ## Adayın deneyim yılı (0–10): 10
    ## Adayın teknik puanı (0–100): 2 değerleri verdiğinde anlam karmaşası 
    # oluyordu o nedenle bu şekilde yaptım. 
    labels = []
    for yoexp, techpoint in zip(years_of_experience, technical_point):
        score = (yoexp / 10) * 0.4 + (techpoint / 100) * 0.6  # 0–1 arası skor
        if score < 0.5:
            labels.append(1)  # Unsuccessful
        else:
            labels.append(0)  # Successful


    technical_point = np.clip(technical_point + np.random.normal(0, 10, n_samples), 0, 100)

    X = np.column_stack((years_of_experience, technical_point))
    y = np.array(labels)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42
    )

    # 🔍 Linear kernel için sadece C optimize ediliyor
    param_dist = {
        'C': uniform(0.01, 100),
        'kernel': ['linear']
    }

    random_search = RandomizedSearchCV(
        SVC(),
        param_distributions=param_dist,
        n_iter=20,
        cv=5,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_

    print("\n En iyi hiperparametreler:", best_params)

    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n Model Doğruluğu: {accuracy * 100:.2f}%")

    print("\n Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\n Classification Report:")
    print(classification_report(y_test, y_pred))

    joblib.dump(best_model, "svm_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("\n Model ve scaler dosyaları kaydedildi.")

    return best_model, scaler, X_scaled, y, y_test, y_pred

model, scaler, X_scaled, y, y_test, y_pred = train_and_save_svm_model_linear()


## Plotting the decision boundary
def plot_decision_boundary(model, X, y,save_path=None):
    plt.figure(figsize=(10, 6))
    
    # Sınıflara göre scatter
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='blue', s=60, edgecolors='k', alpha=0.7)
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='red', s=60, edgecolors='k', alpha=0.7)

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 100)
    yy = np.linspace(ylim[0], ylim[1], 100)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    Z = model.decision_function(xy).reshape(XX.shape)
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1],
               linestyles=['--', '-', '--'])

    # Destek vektörleri
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
               s=150, linewidth=1.5, facecolors='none', edgecolors='k')

    plt.title("SVM Decision Boundary: Candidate Success Classification")
    plt.xlabel("Experience (Years)")
    plt.ylabel("Technical Score (0-100)")
    plt.grid(True)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.text(x=1.05, y=-2.1,
             s="Blue Candidates , \nRed Candidates",
             fontsize=10,
             bbox=dict(facecolor='white', alpha=0.7))

    #Görseli kaydet
    plt.savefig(save_path, dpi=300)
    print(f"Save figure: {save_path}")
    plt.show()

plot_decision_boundary(model, X_scaled, y, save_path=r"C:\GYK\GYK1\ML\SVM_ODEV\karar_siniri.png")


# Evaluating the results
# Get input from the user and make a prediction
try:
    print("\n🔍 Şimdi bir adayın başarı durumunu tahmin edelim.")
    yoexp_input = int(input("Adayın deneyim yılı (0–10): "))
    techpoint_input = int(input("Adayın teknik puanı (0–100): "))

    # Scale user inputs
    new_data = np.array([[yoexp_input, techpoint_input]])
    new_data_scaled = scaler.transform(new_data)

    prediction_index = model.predict(new_data_scaled)[0]
    true_class = model.classes_[prediction_index]  # En doğru yol

    # Results
    print("\nPrediction Result:")
    if true_class == 0:
        print("This candidate was classified as PASSING.")
    else:
        print("This candidate has been classified as FAILED.")

except Exception as e:
    print(f"An error occurred: {e}")
