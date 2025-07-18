import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import os
import pickle
from sklearn.metrics import mean_squared_error
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil.relativedelta import relativedelta

# ------------------- Configuración -------------------
CHECKPOINTS_DIR = Path(r"C:\Users\Martin\OneDrive\Maestría\20- Laboratorio3\RepositorioLabo3\checkpoints")
DB_PATH = CHECKPOINTS_DIR / "optuna_lgbm_study.db"
STUDY_NAME = "lgbm_optim_study"
FUTURE_TARGET = 'tn_future'

# ------------------- Métrica personalizada -------------------
class AbsolutePercentageErrorOnProductTotal:
    def __init__(self, df_eval, product_id_col='product_id', target_col=FUTURE_TARGET):
        self.df_eval = df_eval.copy()
        self.product_id_col = product_id_col
        self.target_col = target_col
        self.metric_name = 'AbsPercErrProdTotal'

    def __call__(self, preds, dataset):
        if not isinstance(preds, np.ndarray):
            preds = np.array(preds)
        labels = dataset.get_label()
        
        df_temp = pd.DataFrame({
            self.product_id_col: self.df_eval[self.product_id_col].values,
            'preds': preds.astype(float),
            'labels': labels.astype(float)
        })

        por_producto = df_temp.groupby(self.product_id_col, observed=True).agg(
            {'labels': 'sum', 'preds': 'sum'}
        )

        sum_total_real_global = por_producto['labels'].sum()
        sum_abs_errors_per_product = np.sum(np.abs(por_producto['labels'] - por_producto['preds']))

        if sum_total_real_global == 0:
            metric_value = 0 if sum_abs_errors_per_product == 0 else sum_abs_errors_per_product
        else:
            metric_value = sum_abs_errors_per_product / sum_total_real_global * 100

        return self.metric_name, metric_value, False

# ------------------- Utilidades -------------------
def load_artifact(name):
    path = CHECKPOINTS_DIR / name
    if (path.with_suffix('.parquet')).exists():
        return pd.read_parquet(path.with_suffix('.parquet'))
    elif (path.with_suffix('.pkl')).exists():
        with open(path.with_suffix('.pkl'), 'rb') as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(f"No se encontró el archivo para el artifact '{name}'.")

def save_artifact(name, data):
    path = CHECKPOINTS_DIR / name
    if isinstance(data, pd.DataFrame):
        data.to_parquet(path.with_suffix('.parquet'), compression='gzip')
    elif isinstance(data, np.ndarray):
        np.save(path.with_suffix('.npy'), data)
    else:
        with open(path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(data, f)

# ------------------- Cargar datos -------------------
X_train = load_artifact("X_train")
y_train = load_artifact("y_train")
X_val = load_artifact("X_val")
y_val = load_artifact("y_val")
X_predict = load_artifact("X_predict")
df_predict = load_artifact("df_predict")
df_val = load_artifact("df_val")
X_test = load_artifact("X_test")
df_test = load_artifact("df_test")
y_test = load_artifact("y_test")

# ------------------- Detección automática de variables categóricas -------------------
def get_categorical_columns(df):
    """Identifica columnas categóricas basándose en su tipo de dato"""
    cat_cols = []
    for col in df.columns:
        if isinstance(df[col].dtype, pd.CategoricalDtype) or \
           pd.api.types.is_object_dtype(df[col]) or \
           pd.api.types.is_string_dtype(df[col]):
            cat_cols.append(col)
    return cat_cols

# Usar X_train como referencia para las categóricas
CATEGORICAL_COLS = get_categorical_columns(X_train)
print(f"\n🔍 Columnas categóricas detectadas: {CATEGORICAL_COLS}")

# Aplicar conversión categórica a TODOS los datasets
for df in [X_train, X_val, X_predict, X_test]:
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype('category')

# ------------------- Optuna objective -------------------
def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 512),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'lambda_l1': trial.suggest_float('lambda_l1', 0, 5.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 0, 5.0),
    }

    custom_metric = AbsolutePercentageErrorOnProductTotal(df_eval=df_val)

    lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=CATEGORICAL_COLS)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train, categorical_feature=CATEGORICAL_COLS)

    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_val],
        valid_names=["val"],
        feval=custom_metric,
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )

    preds = model.predict(X_val, num_iteration=model.best_iteration)
    mse = mean_squared_error(y_val, preds)
    rmse = np.sqrt(mse)
    return rmse

# ------------------- Optuna + entrenamiento final -------------------
try:
    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction='minimize',
        storage=f"sqlite:///{DB_PATH}",
        load_if_exists=True
    )
except:
    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction='minimize',
        storage=f"sqlite:///{DB_PATH}"
    )

# Ejecutar hasta completar 30 trials exitosos
completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
target_trials = 30
max_attempts = 50
attempts = 0

print(f"📊 Trials existentes: {completed_trials} (objetivo: {target_trials})")

while completed_trials < target_trials and attempts < max_attempts:
    attempts += 1
    try:
        study.optimize(objective, n_trials=1, gc_after_trial=True)
        if study.trials[-1].state == optuna.trial.TrialState.COMPLETE:
            completed_trials += 1
            print(f"✅ Trial {completed_trials}/{target_trials} completado")
        else:
            print(f"⏩ Trial fallido (intento {attempts})")
    except Exception as e:
        print(f"⚠️ Error en trial: {str(e)}")

if completed_trials < target_trials:
    print(f"\n⚠️ Completados {completed_trials}/{target_trials} trials (límite de intentos alcanzado)")
else:
    print("\n✨ Todos los trials completados exitosamente")

# ------------------- Entrenar modelo final -------------------
model_path = CHECKPOINTS_DIR / "model.pkl"  # Ajusta extensión si usas .parquet

if model_path.exists():
    print("\n🔍 Cargando modelo existente...")
    final_model = load_artifact("model")
else:
    print("\n🏗️ Entrenando modelo final con los mejores hiperparámetros...")
    best_params = study.best_trial.params
    best_params.update({
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt'
    })
    
    final_model = lgb.train(
        best_params,
        lgb.Dataset(X_train, label=y_train, categorical_feature=CATEGORICAL_COLS),
        valid_sets=[lgb.Dataset(X_val, label=y_val, categorical_feature=CATEGORICAL_COLS)],
        feval=AbsolutePercentageErrorOnProductTotal(df_val),
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )
    save_artifact("model", final_model)
    print("💾 Modelo guardado en:", model_path)

# ------------------- Predicción y evaluación -------------------

# ------------------- Feature importances -------------------
print("\n📈 Generando gráfico de importancias...")

# Calcular importancias
importances = final_model.feature_importance(importance_type='gain')
feature_names = final_model.feature_name()
fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
fi_df = fi_df.sort_values(by='importance', ascending=False)

# Generar y guardar el gráfico (sólo si no existe)
fi_path = CHECKPOINTS_DIR / "feature_importances.png"

if not fi_path.exists():  # <-- Nueva verificación
    plt.figure(figsize=(10, 8))
    sns.barplot(
        data=fi_df.head(30),
        x='importance',
        y='feature',
        hue='feature',
        palette='viridis',
        legend=False
    )
    plt.title("Top 30 Feature Importances (gain)")
    plt.tight_layout()
    plt.savefig(fi_path)
    plt.close()
    print(f"✅ Feature importances guardadas como imagen: {fi_path}")
else:
    print(f"⏩ El archivo {fi_path} ya existe, se omite generación")  # <-- Mensaje útil

def guardar_csv_predicciones(nombre, X, df_base, y_real=None):
    df = df_base.copy()
    df["tn_predicha"] = final_model.predict(X)
    df["fecha_target"] = df["fecha"] + pd.DateOffset(months=2)

    columnas = ["customer_id", "product_id", "fecha", "tn", "tn_predicha", "fecha_target"]

    if y_real is not None:
        df["tn_real_futuro"] = y_real.values
        df["abs_error"] = np.abs(df["tn_real_futuro"] - df["tn_predicha"])
        columnas += ["tn_real_futuro", "abs_error"]

    df_salida = df[columnas]
    output_path = CHECKPOINTS_DIR / f"{nombre}_predictions.csv"
    df_salida.to_csv(output_path, index=False)
    print(f"✅ {nombre}_predictions.csv guardado: {output_path}")

# Generar CSVs para val y test
guardar_csv_predicciones("val", X_val, df_val, y_val)
guardar_csv_predicciones("test", X_test, df_test, y_test[FUTURE_TARGET])


print("\n🎯 ¡Proceso completado!")


# ------------------- Predicción y submission -------------------
print("\n📦 Generando submission...")

y_pred = final_model.predict(X_predict)
df_predict = df_predict.copy()
df_predict["tn_predicha"] = y_pred

submission = df_predict.groupby("product_id")["tn_predicha"].sum().reset_index()
submission.columns = ["product_id", "tn"]

save_artifact("submission", submission)
submission_path = CHECKPOINTS_DIR / "submission.csv"
submission.to_csv(submission_path, index=False)

print(f"✅ Submission guardada como 'submission' y también en: {submission_path}")


#---------------------------- Evaluación de resultados -------------------

import pandas as pd
import numpy as np
import os

# 📊 Clase de métrica
class AbsolutePercentageErrorOnProductTotal:
    def __init__(self, df_eval, product_id_col='product_id', target_col='tn_real_futuro'):
        self.df_eval = df_eval.copy()
        self.product_id_col = product_id_col
        self.target_col = target_col
        self.metric_name = 'AbsPercErrProdTotal'

    def __call__(self, preds, labels):
        preds = np.asarray(preds, dtype=float)
        labels = np.asarray(labels, dtype=float)

        df_temp = pd.DataFrame({
            self.product_id_col: self.df_eval[self.product_id_col].values,
            'preds': preds,
            'labels': labels
        })

        por_producto = df_temp.groupby(self.product_id_col, observed=True).agg(
            {'labels': 'sum', 'preds': 'sum'}
        )

        sum_total_real_global = por_producto['labels'].sum()
        sum_abs_errors_per_product = np.abs(por_producto['labels'] - por_producto['preds']).sum()

        if sum_total_real_global == 0:
            metric_value = 0 if sum_abs_errors_per_product == 0 else sum_abs_errors_per_product
        else:
            metric_value = sum_abs_errors_per_product / sum_total_real_global * 100

        return self.metric_name, metric_value, False

# 📁 Ruta donde se guardaron los archivos
output_dir = r"C:\Users\Martin\OneDrive\Maestría\20- Laboratorio3\RepositorioLabo3\checkpoints"  # 🔁 CAMBIAR por tu ruta real

# 📂 Archivos a evaluar
archivos = {
    "val": os.path.join(output_dir, "val_predictions.csv"),
    "test": os.path.join(output_dir, "test_predictions.csv")
}

resultados = {}

for tipo, ruta in archivos.items():
    print(f"📂 Evaluando: {ruta}")
    try:
        df = pd.read_csv(ruta)

        if not all(col in df.columns for col in ['product_id', 'tn_predicha', 'tn_real_futuro']):
            print(f"❌ Faltan columnas necesarias en {ruta}")
            continue

        metrica = AbsolutePercentageErrorOnProductTotal(df)
        _, valor, _ = metrica(df['tn_predicha'], df['tn_real_futuro'])
        valor = round(valor, 2)
        resultados[tipo] = valor

    except Exception as e:
        print(f"⚠️ Error al procesar {ruta}: {e}")

# 📋 Mostrar resultados
print("\n✅ Resultado final:")
for tipo in ["val", "test"]:
    if tipo in resultados:
        print(f"📊 {tipo.capitalize()} APE Total: {resultados[tipo]} %")
    else:
        print(f"❌ No se obtuvo resultado para {tipo}")
