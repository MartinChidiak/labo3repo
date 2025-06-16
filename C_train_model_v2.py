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
CHECKPOINTS_DIR = os.path.join(GCS_BUCKET_PATH, 'checkpoints2')

# Update to use the final featured checkpoints from pipeline.py
DF_TRAIN_FINAL_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, 'df_train_final_featured.pkl')
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
NUM_LAGS = 12
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

# --- New: Helper functions for modularity ---

def load_processed_data(df_train_path, df_predict_path):
    """Loads the final featured training and prediction dataframes."""
    print("Step 1: Loading final featured data...")
    df_train_fe = load_dataframe_checkpoint(df_train_path)
    df_predict_fe_initial = load_dataframe_checkpoint(df_predict_path)

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

    # Ensure feature columns are identical and in the same order
    if not X_train.columns.equals(X_predict.columns):
        print("Warning: Feature columns do not match between train and predict sets. Reindexing X_predict.")
        # Reindex X_predict to match X_train columns, filling missing with 0 or appropriate default
        X_predict = X_predict.reindex(columns=X_train.columns, fill_value=0) # Consider fill_value=0 or X_train.mean() as appropriate

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
    """Runs Optuna hyperparameter optimization."""
    print("\nStep 4: Starting Optuna hyperparameter optimization...")
    if X_train_eval.empty or X_val_eval.empty:
        print("Skipping Optuna optimization due to insufficient evaluation data.")
        return None

    custom_metric_eval = AbsolutePercentageErrorOnProductTotal(df_val_eval_full, product_id_col='product_id', target_col=future_target_col)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'num_leaves': trial.suggest_int('num_leaves', 31, 128),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 60),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.9, 1.0),
            'random_state': 42,
            'objective': 'tweedie'
        }

        model = lgb.LGBMRegressor(**params)
        model.fit(X_train_eval, y_train_eval,
                  categorical_feature=eval_categorical_cols,
                  eval_set=[(X_val_eval, y_val_eval)],
                  eval_metric=custom_metric_eval,
                  callbacks=[lgb.early_stopping(10, verbose=False)])

        # Calculate the metric value for Optuna's objective (needs to be consistent with eval_metric)
        predictions_val = model.predict(X_val_eval)
        
        # Use the custom metric logic on the predictions to return objective value for Optuna
        # Note: We can reuse the instance or re-instantiate, but the calculation logic must match.
        # Here, we re-create the df_temp similar to the custom_metric_eval's internal logic.
        df_temp = df_val_eval_full.copy() # Use the full validation dataframe slice
        df_temp['preds'] = predictions_val
        df_temp['labels'] = y_val_eval

        por_producto = df_temp.groupby(custom_metric_eval.product_id_col, observed=True).agg(
            {'labels': 'sum', 'preds': 'sum'}
        )
        sum_total_real_global = por_producto['labels'].sum()
        sum_abs_errors_per_product = np.sum(np.abs(por_producto['labels'] - por_producto['preds']))

        if sum_total_real_global == 0:
             objective_value = 0 if sum_abs_errors_per_product == 0 else sum_abs_errors_per_product
        else:
             objective_value = sum_abs_errors_per_product / sum_total_real_global * 100

        return objective_value

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=5) # Adjust n_trials or timeout as needed

    print("\nOptuna optimization finished. Best trial:")
    print(f"  Value: {study.best_trial.value:.4f}")
    print("  Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    
    return study.best_trial.params

def train_final_lgbm_model(X_train, y_train, best_params, categorical_features_names, model_save_path, last_historical_period):
    """Trains the final LightGBM model on the full training data."""
    print(f"\nStep 5: Training final model with best parameters on full historical data up to {last_historical_period}...")
    if X_train.empty or y_train.empty:
        print("Skipping final model training due to empty training data.")
        return None

    lgbm_final = lgb.LGBMRegressor(**best_params)
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

def main_training_script():
    print("Starting Model Training and Prediction Script")

    # Load data
    df_train_fe, df_predict_fe_initial = load_processed_data(DF_TRAIN_FINAL_CHECKPOINT, DF_PREDICT_FINAL_CHECKPOINT)
    if df_train_fe is None or df_predict_fe_initial is None:
        return # Exit if data loading fails

    # Prepare datasets
    X_train, y_train, X_predict, categorical_features_names = prepare_datasets(df_train_fe, df_predict_fe_initial, FUTURE_TARGET, TARGET)

    # Try loading existing model first
    lgbm_final = load_model_checkpoint(FINAL_MODEL_CHECKPOINT)

    # If model not loaded, proceed with optimization and training
    if lgbm_final is None:
        # Create evaluation splits
        X_train_eval, y_train_eval, X_val_eval, y_val_eval, df_val_eval_full, eval_categorical_cols = \
            create_time_based_evaluation_splits(df_train_fe, X_train, y_train, 
                                                validation_start_period=201907, # Hardcoded as in original
                                                last_hist_period=LAST_HISTORICAL_PERIOD, 
                                                target_shift=TARGET_SHIFT)

        # Run Optuna optimization
        best_params = run_optuna_optimization(X_train_eval, y_train_eval, X_val_eval, y_val_eval, 
                                                df_val_eval_full, eval_categorical_cols, FUTURE_TARGET)

        if best_params:
            # Train final model
            lgbm_final = train_final_lgbm_model(X_train, y_train, best_params, categorical_features_names, FINAL_MODEL_CHECKPOINT, LAST_HISTORICAL_PERIOD)
        else:
            print("\nOptimization did not yield best parameters or was skipped. Cannot train final model.")

    # Generate feature importance plot (if model is available)
    if lgbm_final is not None:
        generate_and_save_feature_importance_plot(lgbm_final, X_train, CHECKPOINTS_DIR, PREDICTION_PERIOD, FUTURE_TARGET)

    # Make predictions (if model is available)
    prediction_results_df = make_predictions_and_save_results(lgbm_final, X_predict, df_predict_fe_initial, TARGET, PREDICTION_PERIOD, PREDICTIONS_DF_CHECKPOINT, os.path.join(CHECKPOINTS_DIR, f'predictions_{PREDICTION_PERIOD}.csv'))

    print("\nModel Training and Prediction Script Finished.")


# --- Main Execution ---
if __name__ == "__main__":
    main_training_script()