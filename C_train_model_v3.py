# train_model.py

import pandas as pd
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import pickle # Necesario para guardar/cargar modelos y dataframes
import numpy as np
import optuna # Import Optuna

# Define data paths
GCS_BUCKET_PATH = '/home/chidiakmartin/gcs-bucket'
CHECKPOINTS_DIR = os.path.join(GCS_BUCKET_PATH, 'checkpoints3')

# Update to use the final featured checkpoints from pipeline.py
DF_TRAIN_FINAL_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, 'df_train_final_featured.pkl')  # hasta 201910
DF_PARA_TRAIN_FINAL = os.path.join(CHECKPOINTS_DIR, 'df_para_train_final.pkl')            # hasta 201912
DF_PREDICT_FINAL_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, 'df_predict_final_featured.pkl')


# Define the prediction period (format YYYYMM)
PREDICTION_PERIOD = 202002
# Define the last available historical data period (for training)
LAST_HISTORICAL_PERIOD = 201912
# Define the target variable names
TARGET = 'tn'  # Original target column name
TARGET_SHIFT = 2  # Shift for the future target (t+2)
FUTURE_TARGET = f'{TARGET}_future'  # New column name for the shifted target

# Parameters for feature engineering functions (copied from pipeline.py)
LAG_COLUMNS = ['cust_request_qty', 'cust_request_tn','tn']
NUM_LAGS = 36
# Moving average window (e.g., 3 months)
MOVING_AVG_WINDOW = 3

# Add checkpoints for the trained model and predictions dataframe
FINAL_MODEL_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, 'lgbm_final_model.pkl')
PREDICTIONS_DF_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, f'predictions_df_{PREDICTION_PERIOD}.pkl')

# --- Custom Evaluation Metric (Absolute Percentage Error on Product Total) ---
class AbsolutePercentageErrorOnProductTotal:
    """
    Custom LightGBM evaluation metric calculating the sum of absolute errors
    grouped by product_id, divided by the total real value.
    Formula: SUM(|SUM(Real per Product) - SUM(Pred per Product)| for all Products) / SUM(Total Real Globally)
    """
    def __init__(self, df_eval, product_id_col='product_id', target_col=FUTURE_TARGET):
        self.df_eval = df_eval.copy()
        self.product_id_col = product_id_col
        self.target_col = target_col
        self.metric_name = 'AbsPercErrProdTotal'

    def __call__(self, labels, preds):
        """
        Calculates the metric for LightGBM.

        Args:
            labels (np.ndarray): True labels.
            preds (np.ndarray): Predictions from the model.

        Returns:
            tuple: (metric_name, metric_value, is_higher_better)
        """
        # Create a temporary DataFrame with the predictions and labels
        df_temp = self.df_eval.copy()
        df_temp['preds'] = preds
        df_temp['labels'] = labels

        # Agrupar por product_id y calcular las sumas - Añadimos observed=True
        por_producto = df_temp.groupby(self.product_id_col, observed=True).agg(
            {'labels': 'sum', 'preds': 'sum'}
        )

        # Calcular la suma total real global
        sum_total_real_global = por_producto['labels'].sum()

        # Calcular el numerador: suma de errores absolutos de las sumas por producto
        sum_abs_errors_per_product = np.sum(np.abs(por_producto['labels'] - por_producto['preds']))

        # Calcular la métrica
        if sum_total_real_global == 0:
            metric_value = 0 if sum_abs_errors_per_product == 0 else sum_abs_errors_per_product
        else:
            metric_value = sum_abs_errors_per_product / sum_total_real_global * 100

        return self.metric_name, metric_value, False


# --- Model Training and Prediction Steps ---

def train_lgbm_model(X_train, y_train, categorical_features_names):
    """Trains the LightGBM Regressor model using native categorical feature handling."""
    print("Training LightGBM model...")

    lgbm = lgb.LGBMRegressor(random_state=42) # Use a fixed random state for reproducibility

    # Train the model, passing the names of the categorical features
    lgbm.fit(X_train, y_train,
             categorical_feature=categorical_features_names,
             # Add silent=True to suppress verbose training output if needed
             # callbacks=[lgb.early_stopping(10, verbose=False)] # Example for early stopping if you have validation data here
             )

    print("Model training complete.")
    return lgbm

def save_model_checkpoint(model, path):
    """Saves the trained LightGBM model to a file."""
    print(f"Saving model checkpoint to {path}")
    try:
        # Use pickle for consistency with DataFrame saving
        with open(path, 'wb') as f:
             pickle.dump(model, f)
        print("Model checkpoint saved successfully.")
    except Exception as e:
        print(f"Error saving model checkpoint: {e}")

def load_model_checkpoint(path):
    """Loads a trained LightGBM model from a file."""
    if os.path.exists(path):
        print(f"Loading model checkpoint from {path}")
        try:
            # Use pickle for consistency with model saving
            with open(path, 'rb') as f:
                 lgbm = pickle.load(f)

            print("Model checkpoint loaded successfully.")
            return lgbm
        except Exception as e:
            print(f"Error loading model checkpoint: {e}")
            return None # Return None if loading fails
    else:
        print(f"Model checkpoint not found at {path}")
        return None

def save_dataframe_checkpoint(df, path):
    """Saves a DataFrame to a pickle file."""
    print(f"Saving DataFrame checkpoint to {path}")
    try:
        df.to_pickle(path)
        print("DataFrame checkpoint saved successfully.")
    except Exception as e:
        print(f"Error saving DataFrame checkpoint: {e}")


def load_dataframe_checkpoint(path):
    """Loads a DataFrame from a pickle file."""
    if os.path.exists(path):
        print(f"Loading DataFrame checkpoint from {path}")
        try:
            df = pd.read_pickle(path)
            print("DataFrame checkpoint loaded successfully.")
            return df
        except Exception as e:
            print(f"Error loading DataFrame checkpoint: {e}")
            return None # Return None if loading fails
    else:
        print(f"DataFrame checkpoint not found at {path}")
        return None

df_train_fe_full = load_dataframe_checkpoint(DF_PARA_TRAIN_FINAL)

# --- New: Helper functions for modularity ---

def load_processed_data(df_train_path, df_predict_path):
    """Loads the final featured training and prediction dataframes."""
    print("Step 1: Loading final featured data...")
    df_train_fe = load_dataframe_checkpoint(df_train_path)
    df_predict_fe_initial = load_dataframe_checkpoint(df_predict_path)
    df_train_fe_full = load_dataframe_checkpoint(DF_PARA_TRAIN_FINAL)
    
    if df_train_fe is None or df_predict_fe_initial is None:
        print("Error: Could not load featured dataframes. Ensure pipeline.py ran successfully.")
        return None, None
    print("Dataframes loaded successfully.")
    return df_train_fe, df_predict_fe_initial

def prepare_datasets(df_train_fe, df_predict_fe_initial, target_future_col, target_original_col):
    """Identifies features and target, and prepares datasets for modeling."""
    print("Step 2: Preparing datasets for training and prediction...")
    if target_future_col not in df_train_fe.columns:
        raise ValueError(f"Target column '{target_future_col}' not found in training data. Ensure pipeline.py created it.")
    y_train = df_train_fe[target_future_col]

    id_cols = ['customer_id', 'product_id', 'periodo']
    date_cols = ['fecha'] # Exclude 'fecha' from features
    target_cols = [target_original_col, target_future_col]

    # --- NUEVO: Forzar tipo category en columnas explícitas ---
    categorical_cols_explicit = [
        'customer_id', 'product_id', 'periodo', 'cat1', 'cat2', 'cat3', 'brand', 'cliente_categoria', 'is_macro_event'
        # agrega aquí cualquier otra columna que quieras tratar como categórica
    ]
    for col in categorical_cols_explicit:
        if col in df_train_fe.columns:
            df_train_fe[col] = df_train_fe[col].astype('category')
        if col in df_predict_fe_initial.columns:
            df_predict_fe_initial[col] = df_predict_fe_initial[col].astype('category')

    # Identify categorical columns that are already 'category' dtype from pipeline.py
    # and are not target or ID columns that we'll exclude
    categorical_features_names = [col for col in df_train_fe.columns 
                                  if df_train_fe[col].dtype == 'category' 
                                  and col not in id_cols + target_cols]

    # All columns excluding IDs, original fecha, original target, and future target
    features = [col for col in df_train_fe.columns 
                if col not in id_cols + date_cols + target_cols + ['cliente_categoria']]

    X_train = df_train_fe[features].copy()
    X_predict = df_predict_fe_initial[features].copy()

    # --- Asegura tipos y categorías ---
    for col in categorical_features_names:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype('category')
        if col in X_predict.columns:
            X_predict[col] = X_predict[col].astype('category')
            # Alinea categorías de predict con las de train
            X_predict[col] = X_predict[col].cat.set_categories(X_train[col].cat.categories)

    # Elimina columnas categóricas que no estén en ambos sets
    categorical_features_names = [col for col in categorical_features_names if col in X_train.columns and col in X_predict.columns]

    # Ensure feature columns are identical and in the same order
    if not X_train.columns.equals(X_predict.columns):
        print("Warning: Feature columns do not match between train and predict sets. Reindexing X_predict.")
        X_predict = X_predict.reindex(columns=X_train.columns, fill_value=0)

    print(f"Features identified: {len(features)} columns.")
    print(f"Categorical features: {categorical_features_names}")
    return X_train, y_train, X_predict, categorical_features_names

def create_time_based_evaluation_splits(df_train_fe, X_train, y_train, validation_start_period, last_hist_period, target_shift):
    """Creates time-based evaluation splits for hyperparameter tuning."""
    print(f"\nStep 3: Creating time-based evaluation splits for Optuna...")
    # Ensure fecha is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_train_fe['fecha']):
        df_train_fe['fecha'] = pd.to_datetime(df_train_fe['fecha'])

    val_start_date = pd.to_datetime(str(validation_start_period), format='%Y%m')
    val_end_date = pd.to_datetime(str(last_hist_period), format='%Y%m') - pd.DateOffset(months=target_shift)

    print(f"  Evaluation Train: data before {val_start_date.strftime('%Y%m')}")
    print(f"  Evaluation Val: data from {val_start_date.strftime('%Y%m')} to {val_end_date.strftime('%Y%m')}")

    # Use boolean indexing for cleaner and potentially more efficient splitting
    train_eval_mask = df_train_fe['fecha'] < val_start_date
    val_eval_mask = (df_train_fe['fecha'] >= val_start_date) & (df_train_fe['fecha'] <= val_end_date)

    X_train_eval = X_train[train_eval_mask].copy()
    y_train_eval = y_train[train_eval_mask].copy()
    X_val_eval = X_train[val_eval_mask].copy()
    y_val_eval = y_train[val_eval_mask].copy()
    
    # Pass the relevant slice of the original df_train_fe for the custom metric
    df_val_eval_full = df_train_fe[val_eval_mask].copy()

    print(f"  Evaluation Training data shape: {X_train_eval.shape}")
    print(f"  Evaluation Validation data shape: {X_val_eval.shape}")
    
    # Filter categorical columns to only those present in the evaluation split (important for LightGBM)
    eval_categorical_cols = [col for col in X_train.columns if col in X_train_eval.columns and X_train_eval[col].dtype == 'category']

    return X_train_eval, y_train_eval, X_val_eval, y_val_eval, df_val_eval_full, eval_categorical_cols

def run_optuna_optimization(X_train_eval, y_train_eval, X_val_eval, y_val_eval, df_val_eval_full, eval_categorical_cols, future_target_col):
    print("\nStep 4: Starting Optuna hyperparameter optimization...")

    if X_train_eval.empty or X_val_eval.empty:
        print("Skipping Optuna optimization due to insufficient evaluation data.")
        return None

    custom_metric_eval = AbsolutePercentageErrorOnProductTotal(df_val_eval_full, product_id_col='product_id', target_col=future_target_col)

    def objective(trial):
        # Ampliamos el espacio de búsqueda basándonos en las recomendaciones.
        params = {
            'objective': 'tweedie',
            'random_state': 42,
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 31, 256),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        }

        model = lgb.LGBMRegressor(**params, n_jobs=-1)
        model.fit(X_train_eval, y_train_eval,
                  categorical_feature=eval_categorical_cols,
                  eval_set=[(X_val_eval, y_val_eval)],
                  eval_metric=custom_metric_eval,
                  callbacks=[lgb.early_stopping(10, verbose=False)])

        # Forma más limpia y eficiente de obtener el score de la métrica personalizada
        return model.best_score_['valid_0'][custom_metric_eval.metric_name]

    # --- NUEVO: Checkpointing con SQLite ---
    optuna_db_path = os.path.join(CHECKPOINTS_DIR, "optuna_study.db")
    storage_url = f"sqlite:///{optuna_db_path}"
    study = optuna.create_study(
        direction='minimize',
        storage=storage_url,
        study_name="lgbm_hyperopt",
        load_if_exists=True
    )

    # --- Check for completed trials before running new ones ---
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"Found {len(completed_trials)} completed trials in the study.")

    total_trials_target = 30  # The desired total number of completed trials

    if len(completed_trials) >= total_trials_target:
        print(f"Skipping optimization as {len(completed_trials)} trials are already complete (target: {total_trials_target}).")
    else:
        remaining_trials = total_trials_target - len(completed_trials)
        print(f"Continuing optimization, running up to {remaining_trials} more trials...")
        study.optimize(objective, n_trials=remaining_trials)


    print("\nOptuna optimization finished. Best trial:")
    print(f"  Value: {study.best_trial.value:.4f}")
    print("  Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

    # (Opcional) Guarda los mejores parámetros
    with open(os.path.join(CHECKPOINTS_DIR, "optuna_best_params.pkl"), "wb") as f:
        pickle.dump(study.best_trial.params, f)

    return study.best_trial.params

def train_final_lgbm_model(X_train, y_train, best_params, categorical_features_names, model_save_path, last_historical_period):
    """Trains the final LightGBM model on the full training data."""
    print(f"\nStep 5: Training final model with best parameters on full historical data up to {last_historical_period}...")
    if X_train.empty or y_train.empty:
        print("Skipping final model training due to empty training data.")
        return None

    lgbm_final = lgb.LGBMRegressor(**best_params, n_jobs=-1)
    lgbm_final.fit(X_train, y_train, categorical_feature=categorical_features_names)
    save_model_checkpoint(lgbm_final, model_save_path)
    return lgbm_final

def generate_and_save_feature_importance_plot(model, features_df, output_dir, prediction_period, target_col):
    """Generates and saves the feature importance plot."""
    if model is None:
        print("Skipping feature importance plot: Model not trained.")
        return

    print("\nStep 6: Generating feature importance plot...")
    feature_names = model.feature_name_ if hasattr(model, 'feature_name_') else features_df.columns.tolist()
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    importance = importance.sort_values('importance', ascending=False).head(40)

    plt.figure(figsize=(10, 7))
    ax = sns.barplot(x='importance', y='feature', data=importance, palette='viridis')
    plt.title(f'Top 40 Feature Importance (Predicting {prediction_period})', fontsize=16)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)

    for p in ax.patches:
        width = p.get_width()
        format_string = '{:1.4f}' if width < 0.01 else '{:1.2f}'
        plt.text(width + 0.005, p.get_y() + p.get_height()/2.,
                format_string.format(width),
                ha='left', va='center', fontsize=10)

    plt.tight_layout()
    importance_plot_path = os.path.join(output_dir, f'feature_importance_{prediction_period}.png')
    plt.savefig(importance_plot_path, bbox_inches='tight')
    print(f"Feature importance plot saved to {importance_plot_path}")

def make_predictions_and_save_results(model, X_predict, df_predict_initial, target_original_col, prediction_period, predictions_df_path, predictions_csv_path):
    """Makes predictions and saves the results to pickle and CSV."""
    if model is None:
        print("Skipping predictions: Model not trained or loaded.")
        return None

    print(f"\nStep 7: Making predictions for period {prediction_period}...")
    predictions_predict = model.predict(X_predict)

    prediction_results_df = df_predict_initial[['customer_id', 'product_id', 'fecha', target_original_col]].copy()
    prediction_results_df[f'{target_original_col}_predicted_{prediction_period}'] = predictions_predict
    save_dataframe_checkpoint(prediction_results_df, predictions_df_path)

    if prediction_results_df is not None and not prediction_results_df.empty:
        prediction_results_df.to_csv(predictions_csv_path, index=False)
        print(f"Predictions saved to {predictions_csv_path}")
        print("\nPrediction Examples:")
        print(prediction_results_df.head())
    
    return prediction_results_df

def train_and_predict_with_seeds(X_train, y_train, X_predict, categorical_features_names, best_params, seeds, model_save_dir=None):
    """
    Entrena un modelo LightGBM por cada semilla, guarda cada modelo y promedia las predicciones.
    Si el modelo ya existe, lo carga y predice.
    """
    predictions_list = []
    loaded_seeds = []
    for seed in seeds:
        model_path = os.path.join(model_save_dir, f"lgbm_final_model_seed_{seed}.pkl") if model_save_dir is not None else None
        model = None
        # Si el modelo ya existe, cargarlo
        if model_path is not None and os.path.exists(model_path):
            print(f"Modelo para seed {seed} ya existe. Cargando...")
            model = load_model_checkpoint(model_path)
        # Si no existe, entrenar y guardar
        if model is None:
            print(f"Entrenando modelo con semilla {seed}...")
            params = best_params.copy()
            params['random_state'] = seed
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train, categorical_feature=categorical_features_names)
            if model_save_dir is not None:
                save_model_checkpoint(model, model_path)
        preds = model.predict(X_predict)
        predictions_list.append(preds)
        loaded_seeds.append(seed)
    # Promediar las predicciones
    predictions_mean = np.mean(predictions_list, axis=0)
    return predictions_mean, np.array(predictions_list), loaded_seeds

def load_best_params(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        print(f"Best params file not found at {path}")
        return None

def generate_validation_outputs_if_needed(
    lgbm_final, X_val_eval, y_val_eval, df_val_eval_full, checkpoints_dir
):
    csv_cliente_producto = os.path.join(checkpoints_dir, 'validacion_preds_cliente_producto.csv')
    csv_por_producto = os.path.join(checkpoints_dir, 'validacion_preds_por_producto.csv')

    # Si ambos archivos ya existen, no hace falta volver a generarlos
    if os.path.exists(csv_cliente_producto) and os.path.exists(csv_por_producto):
        print("Los archivos de validación ya existen. Saltando generación de CSVs.")
        return

    # Si no existen, generarlos
    y_val_pred = lgbm_final.predict(X_val_eval)
    df_val_preds = df_val_eval_full[['customer_id', 'product_id', 'fecha']].copy()
    df_val_preds['real'] = y_val_eval.values
    df_val_preds['pred'] = y_val_pred
    df_val_preds.to_csv(csv_cliente_producto, index=False)
    print(f"Guardado: {csv_cliente_producto}")

    df_prod_agg = df_val_preds.groupby('product_id').agg(
        real_total=('real', 'sum'),
        pred_total=('pred', 'sum')
    ).reset_index()
    df_prod_agg['abs_error'] = (df_prod_agg['real_total'] - df_prod_agg['pred_total']).abs()
    df_prod_agg['abs_perc_error'] = 100 * df_prod_agg['abs_error'] / df_prod_agg['real_total'].replace(0, np.nan)
    df_prod_agg.to_csv(csv_por_producto, index=False)
    print(f"Guardado: {csv_por_producto}")

def main_training_script():
    print("Starting Model Training and Prediction Script")

    # Load data
    df_train_fe, df_predict_fe_initial = load_processed_data(DF_TRAIN_FINAL_CHECKPOINT, DF_PREDICT_FINAL_CHECKPOINT)
    if df_train_fe is None or df_predict_fe_initial is None:
        return # Exit if data loading fails

    # Prepare datasets
    X_train, y_train, X_predict, categorical_features_names = prepare_datasets(df_train_fe, df_predict_fe_initial, FUTURE_TARGET, TARGET)

    # --- SIEMPRE crear los splits de validación ---
    X_train_eval, y_train_eval, X_val_eval, y_val_eval, df_val_eval_full, eval_categorical_cols = \
        create_time_based_evaluation_splits(df_train_fe, X_train, y_train, 
                                            validation_start_period=201907, # Hardcoded as in original
                                            last_hist_period=LAST_HISTORICAL_PERIOD, 
                                            target_shift=TARGET_SHIFT)

    # Try loading existing model first
    lgbm_final = load_model_checkpoint(FINAL_MODEL_CHECKPOINT)

    # --- NUEVO: Cargar best_params si existen ---
    OPTUNA_BEST_PARAMS_PATH = os.path.join(CHECKPOINTS_DIR, "optuna_best_params.pkl")
    best_params = None
    if os.path.exists(OPTUNA_BEST_PARAMS_PATH):
        best_params = load_best_params(OPTUNA_BEST_PARAMS_PATH)

    # If model not loaded, proceed with optimization and training
    if lgbm_final is None:
        # Run Optuna optimization
        best_params = run_optuna_optimization(X_train_eval, y_train_eval, X_val_eval, y_val_eval, 
                                                df_val_eval_full, eval_categorical_cols, FUTURE_TARGET)

        if best_params:
            # Prepara datasets con el dataframe extendido (incluye 201911 y 201912)
            X_train_full, y_train_full, _, categorical_features_names_full = prepare_datasets(
                df_train_fe_full, df_predict_fe_initial, FUTURE_TARGET, TARGET
            )

            # Entrena el modelo final con todos los datos históricos
            lgbm_final = train_final_lgbm_model(
                X_train_full, y_train_full, best_params, categorical_features_names_full, FINAL_MODEL_CHECKPOINT, LAST_HISTORICAL_PERIOD
            )
        else:
            print("\nOptimization did not yield best parameters or was skipped. Cannot train final model.")

    # Generate feature importance plot (if model is available)
    if lgbm_final is not None:
        generate_and_save_feature_importance_plot(lgbm_final, X_train, CHECKPOINTS_DIR, PREDICTION_PERIOD, FUTURE_TARGET)

    # Make predictions (if model is available)
    if lgbm_final is not None and best_params is not None:
        seeds = [42, 2024, 7, 123, 999][:5]
        predictions_mean, predictions_array, loaded_seeds = train_and_predict_with_seeds(
            X_train, y_train, X_predict, categorical_features_names, best_params, seeds, model_save_dir=CHECKPOINTS_DIR
        )

        # Guarda el resultado promedio
        prediction_results_df = df_predict_fe_initial[['customer_id', 'product_id', 'fecha', TARGET]].copy()
        prediction_results_df[f'{TARGET}_predicted_{PREDICTION_PERIOD}'] = predictions_mean
        save_dataframe_checkpoint(prediction_results_df, PREDICTIONS_DF_CHECKPOINT)
        prediction_results_df.to_csv(os.path.join(CHECKPOINTS_DIR, f'predictions_{PREDICTION_PERIOD}.csv'), index=False)
        print(f"Predicciones ensemble guardadas usando semillas: {loaded_seeds}")
        print("\nEjemplo de predicciones ensemble:")
        print(prediction_results_df.head())

        # Visualización de la dispersión:
        std_per_sample = np.std(predictions_array, axis=0)
        plt.figure(figsize=(10, 5))
        plt.hist(std_per_sample, bins=30, color='skyblue', edgecolor='black')
        plt.title('Dispersión de las predicciones entre semillas (std por muestra)')
        plt.xlabel('Desviación estándar de la predicción')
        plt.ylabel('Cantidad de muestras')
        plt.tight_layout()
        dispersion_plot_path = os.path.join(CHECKPOINTS_DIR, f'dispersion_predicciones_{PREDICTION_PERIOD}.png')
        plt.savefig(dispersion_plot_path)
        plt.show()
        print(f"Gráfico de dispersión guardado en: {dispersion_plot_path}")

        # Generate validation outputs if needed
        generate_validation_outputs_if_needed(
            lgbm_final, X_val_eval, y_val_eval, df_val_eval_full, CHECKPOINTS_DIR
        )
    else:
        # Fallback: predicción simple si no hay ensemble
        prediction_results_df = make_predictions_and_save_results(
            lgbm_final, X_predict, df_predict_fe_initial, TARGET, PREDICTION_PERIOD,
            PREDICTIONS_DF_CHECKPOINT, os.path.join(CHECKPOINTS_DIR, f'predictions_{PREDICTION_PERIOD}.csv')
        )

    print("\nModel Training and Prediction Script Finished.")


# --- Main Execution ---
if __name__ == "__main__":
    main_training_script()