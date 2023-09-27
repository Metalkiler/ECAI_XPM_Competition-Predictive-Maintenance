import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from pqdm.processes import pqdm as pqdm_processes
from sklearn.cluster import KMeans, Birch
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from tqdm import trange

optimizationType = "Bayesian"
modeltoRun = "RF"
featureChoice = sys.argv[1]  # Air or Oil
unsupervisedMethod = sys.argv[2]  # Kmeans, Iforest, Birch

CPUs = os.cpu_count()
print("Total CPUs count:", CPUs)

df = pd.read_csv("Metro.csv")
_ = df.pop("Unnamed: 0")
df2 = df.convert_dtypes()
df2['timestamp'] = pd.to_datetime(df2['timestamp'], )

datas = df2.pop("timestamp")

if featureChoice == "Air":
    Features = ["TP2", "TP3", "H1", "DV_pressure", "COMP", "Motor_current"]
else:
    Features = ["Oil_temperature", "Oil_level"]

Leaks = df2[Features].copy()
scaler = StandardScaler()
scaled_data = scaler.fit_transform(Leaks)

pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_data)

if unsupervisedMethod == "Kmeans":
    n_clusters = 2  # You can adjust this based on your needs

    # Train the K-Means model
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)


    # Calculate the distance of each data point to the nearest cluster center
    distances = kmeans.fit_transform(principal_components)

    min_distances = np.min(distances, axis=1)

    # Define a threshold for anomaly detection
    anomaly_threshold = np.percentile(min_distances, 95)  # Adjust as needed

    # Identify anomalies based on the threshold
    is_anomaly = min_distances > anomaly_threshold

    # Display the results
    print(f"Anomaly threshold: {anomaly_threshold}")
    print(f"Number of anomalies: {sum(is_anomaly)}")
elif unsupervisedMethod == "Iforest":

    # Train an Isolation Forest model
    isolation_forest = IsolationForest(random_state=0, n_jobs=-1)  # Adjust contamination as needed
    isolation_forest.fit(principal_components)

    # Predict anomalies using the Isolation Forest model
    anomaly_predictions = isolation_forest.predict(principal_components)

    # Convert prediction labels to binary (1 for anomalies, -1 for inliers)
    is_anomaly = anomaly_predictions == -1
    print(f"Number of anomalies: {sum(is_anomaly)}")

elif unsupervisedMethod == "Birch":
    n_clusters = 2
    # Apply BIRCH clustering on the principal components
    birch = Birch(n_clusters=n_clusters)  # Adjust parameters as needed

    # Calculate the distance of each data point to the nearest cluster center
    distances = birch.fit_transform(principal_components)
    min_distances = np.min(distances, axis=1)

    # Define a threshold for anomaly detection
    anomaly_threshold = np.percentile(min_distances, 95)  # Adjust as needed

    # Identify anomalies based on the threshold
    is_anomaly = min_distances > anomaly_threshold

    # Display the results
    print(f"Anomaly threshold: {anomaly_threshold}")
    print(f"Number of anomalies: {sum(is_anomaly)}")

# Add a column to the DataFrame indicating anomalies
Leaks['Target'] = is_anomaly.astype(int)
Checker = pd.concat([Leaks, pd.DataFrame(datas)], axis=1)
# Display the DataFrame with the anomaly detection results
print(Checker)

splitpoint = int(Leaks.shape[0] * 0.7)

tr = Leaks.iloc[0:splitpoint].copy()
ts = Leaks.iloc[splitpoint:].copy()
ytr = tr.pop("Target")
yts = ts.pop("Target")

splitpoint_val = int(tr.shape[0] * 0.7)

tr_val = tr.iloc[0:splitpoint_val].copy()
ytr_val = ytr.iloc[0:splitpoint_val].copy()
ts_val = tr.iloc[splitpoint_val:].copy()
yts_val = ytr.iloc[splitpoint_val:].copy()


def objective(trial):
    paramsRF = {
        # 'max_depth': trial.suggest_int('max_depth', 5, 100),
        'min_samples_split': trial.suggest_float('min_samples_split', 0.01, 1),
        'max_features': trial.suggest_float('max_features', 0.01, 1),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
        'n_estimators': trial.suggest_int('n_estimators', 10, 400),
        'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0, 0.2),
        'oob_score': trial.suggest_categorical('oob_score', [True, False]),
        'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0, 0.5)
    }
    model = RandomForestClassifier(**paramsRF, n_jobs=-1)
    model.fit(tr_val, ytr_val)  # run the classifier on the task

    pred = model.predict_proba(ts_val)[:, 1]
    score = roc_auc_score(yts_val, pred)

    return score


def bestModel(modeltoRun, params):
    model = RandomForestClassifier(**params, n_jobs=-1)
    return model


study_name = "KaggleotimizationBinary_" + featureChoice + "_" + optimizationType + '_' + modeltoRun + "metroPT_with" + unsupervisedMethod
storage = optuna.storages.InMemoryStorage()
study = optuna.create_study(study_name=study_name,
                            sampler=optuna.samplers.TPESampler(multivariate=True),
                            storage=storage, direction='maximize')

study.optimize(objective, n_trials=100)


def anticipation_stoppage(df, A=0, min_group_size=15):
    """
    Neste caso preciso de ter em conta os 150 pontos da anomalia e o inicio do segmento 2

    :param min_group_size: Minimo de tamanho do segmento
    :param df: dataframe com as pontuações e valores verdadeiros para gerar o AUC
    :param A: nivel de antecipação
    :return: AUC de antecipação (sensibilidade)
    """
    df["Segmento"] = None
    lista_paragens = np.where(df.Target.values == 1)[0].tolist()

    sublista = []
    for i in lista_paragens:
        sublista.append(df.iloc[i:].Target.ne(1).idxmax() - 1)

    listaParagensFinais = np.unique(sublista)
    contador = 0
    segmento = 0

    for i in listaParagensFinais:
        segmento = segmento + 1
        if i != listaParagensFinais[-1]:
            df.Segmento.iloc[contador:i] = segmento
        else:
            df.Segmento.iloc[i:] = segmento
        contador = i + 1

    df.Segmento = df.Segmento.fillna(segmento + 1)

    grupos = df.groupby('Segmento')

    listDF = []
    dsSize = 0

    for key, item in grupos:
        listDF.append(key)
        dsSize = dsSize + 1

    if A == 0:
        roc_auc = roc_valuegen_anticipated(df.Target.values.flatten(), df.Probabilidade_STOP.values.flatten(), A)
    else:
        # print("Tamanho de Segmentos (separados por turno e a questão da paragem): ", dsSize)
        trueY, probability_scores = [], []
        for seg in listDF:
            segDF = grupos.get_group(seg).copy()
            sizes = segDF.shape[0]
            if sizes > min_group_size:
                PredVals = segDF.Probabilidade_STOP.values.tolist()
                trueVals = segDF.Target.values.tolist()
                vectorPred = PredVals[:-A]
                vectorTrue = trueVals[A:]
                trueY.extend(vectorTrue)
                probability_scores.extend(vectorPred)

        roc_auc = roc_valuegen_anticipated(trueY, probability_scores, A)
        # print("Minimo Segmentos:", listsizes)
    return roc_auc


def roc_valuegen_anticipated(trueY, probability_scores, A):
    vectorPred = probability_scores
    vectorTrue = trueY
    fpr, tpr, _ = roc_curve(vectorTrue, vectorPred)
    roc_auc = auc(fpr, tpr)
    return roc_auc


joblib.dump(study, study_name + '.pkl')

pipe = bestModel(modeltoRun, study.best_params)
pipe.fit(tr, ytr)
probs = pipe.predict_proba(ts)[:, 1]
probs_df = pd.DataFrame({'Probabilidade_STOP': probs})
Target = pd.DataFrame(yts)
probs_df.index = Target.index
ScoresProb = pd.concat([ts, Target, probs_df], axis=1)
ScoresProb

ScoresProb.reset_index(drop=True, inplace=True)
ScoresProb.to_csv(
    modeltoRun + "_" + optimizationType + "_PCA_with_" + unsupervisedMethod + "_" + featureChoice + ".csv", index=False)

vectorPred = ScoresProb.Probabilidade_STOP.values.flatten()
vectorTrue = ScoresProb.Target.values.flatten()
fpr, tpr, _ = roc_curve(vectorTrue, vectorPred)
roc_auc = auc(fpr, tpr)
roc_auc

# plt.figure()
# lw = 2
# plt.plot(fpr, tpr, color='darkorange',
#          lw=lw, label='Roc Curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('FPR')
# plt.ylabel('TPR')
# plt.legend(loc="lower right")
# plt.show()

# Ahead of Time 720 records ≈ 2 hours


A = 1080
args = [{"df": ScoresProb, "A": i} for i in trange(0, A)]
ModelAUC = pqdm_processes(args, anticipation_stoppage, n_jobs=1, argument_type='kwargs')


def plotAUCAnticipation_single(lista, name1, feature, unsupervisedMethod):
    plt.figure()
    plt.plot(lista, 'r--', label=name1)
    plt.legend()
    plt.xlabel('Anticipation level')
    plt.ylabel('AUC')
    plt.savefig(unsupervisedMethod + "AnticipationGraphics_" + name1 + "_" + feature + ".pdf")
    plt.show()


plotAUCAnticipation_single(ModelAUC, modeltoRun, featureChoice, unsupervisedMethod)

pd.DataFrame(ModelAUC).to_csv("AUC_Up3Hours_RF_" + unsupervisedMethod + "_" + featureChoice + ".csv", index=False)
