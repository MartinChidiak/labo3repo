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

# Import specific feature engineering functions from the pipeline helper script


# Define data paths
CHECKPOINTS_DIR = r'C:\Users\Martin\OneDrive\Maestría\20- Laboratorio3\Data\checkpoints'
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

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting Model Training Script")

    # Step 1: Load the final featured data from pipeline.py
    # Load data that has already been fully featured and processed by pipeline.py
    df_train_fe = load_dataframe_checkpoint(DF_TRAIN_FINAL_CHECKPOINT)
    df_predict_fe_initial = load_dataframe_checkpoint(DF_PREDICT_FINAL_CHECKPOINT)

    # Check if dataframes were loaded successfully
    if df_train_fe is None or df_predict_fe_initial is None:
        print("Error loading featured dataframes. Please ensure pipeline.py has been run successfully.")
        exit() # Exit the script if data loading fails

    # Identify target variable for training - now using FUTURE_TARGET
    if FUTURE_TARGET not in df_train_fe.columns:
        raise ValueError(f"Target column '{FUTURE_TARGET}' not found in training data. Ensure pipeline.py created it.")
    y_train = df_train_fe[FUTURE_TARGET]

    # Identify features (X)
    id_cols = ['customer_id', 'product_id', 'periodo']
    date_cols = ['fecha']
    target_cols = [TARGET, FUTURE_TARGET]  # Exclude both original and future target

    # Get categorical columns (should already be in 'category' dtype from pipeline.py)
    categorical_cols = [col for col in df_train_fe.columns if df_train_fe[col].dtype == 'category'
                       and col not in target_cols]

    # Then exclude the original fecha column from features
    features = [col for col in df_train_fe.columns if col not in ['fecha'] + target_cols + ['cliente_categoria']]

    X_train = df_train_fe[features].copy()
    X_predict = df_predict_fe_initial[features].copy()

    # Ensure feature columns are identical and in the same order
    if not X_train.columns.equals(X_predict.columns):
        print("Warning: Feature columns do not match between train and predict sets.")
        X_predict = X_predict.reindex(columns=X_train.columns, fill_value=0)

    # --- Evaluation Split ---
    # Split based on time, ensuring the target (t+2) is available
    validation_split_period_start = pd.to_datetime('201907', format='%Y%m')
    last_valid_eval_date = pd.to_datetime(str(LAST_HISTORICAL_PERIOD), format='%Y%m') - pd.DateOffset(months=TARGET_SHIFT)

    print(f"\nSplitting historical train data for evaluation (train_eval < {validation_split_period_start.strftime('%Y%m')}, val_eval from {validation_split_period_start.strftime('%Y%m')} up to {last_valid_eval_date.strftime('%Y%m')})...")

    # Ensure fecha is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_train_fe['fecha']):
        df_train_fe['fecha'] = pd.to_datetime(df_train_fe['fecha'])

    # Select indices for evaluation split
    train_eval_indices = df_train_fe[df_train_fe['fecha'] < validation_split_period_start].index
    val_eval_indices = df_train_fe[(df_train_fe['fecha'] >= validation_split_period_start) & 
                                 (df_train_fe['fecha'] <= last_valid_eval_date)].index

    # Create evaluation splits
    X_train_eval = X_train.loc[X_train.index.intersection(train_eval_indices)].copy()
    y_train_eval = y_train.loc[y_train.index.intersection(train_eval_indices)].copy()
    X_val_eval = X_train.loc[X_train.index.intersection(val_eval_indices)].copy()
    y_val_eval = y_train.loc[y_train.index.intersection(val_eval_indices)].copy()

    print(f"Evaluation Training data shape: {X_train_eval.shape}")
    print(f"Evaluation Validation data shape: {X_val_eval.shape}")

    # Ensure category dtypes are correct in evaluation splits
    eval_categorical_cols = [col for col in categorical_cols if col in X_train_eval.columns and X_train_eval[col].dtype == 'category']
    # The conversion to category dtype is now done in pipeline.py, so we just identify the columns present.
    # for col in eval_categorical_cols:
    #     if col in X_train_eval.columns and X_train_eval[col].dtype != 'category':
    #         X_train_eval[col] = X_train_eval[col].astype('category')
    #     if col in X_val_eval.columns and X_val_eval[col].dtype != 'category':
    #         X_val_eval[col] = X_val_eval[col].astype('category')

    # --- Hyperparameter Optimization with Optuna ---

    # Instantiate the custom metric class for Optuna objective and LightGBM eval_metric
    # Pass the full validation DataFrame slice
    custom_metric_eval = AbsolutePercentageErrorOnProductTotal(df_train_fe.loc[df_train_fe.index.intersection(val_eval_indices)].copy(), product_id_col='product_id', target_col=FUTURE_TARGET)

    # Define the Optuna objective function
    def objective(trial):
        # Suggest hyperparameters
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000), # Number of boosting rounds
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2), # Step size shrinkage
            'num_leaves': trial.suggest_int('num_leaves', 31, 128), # Maximum number of leaves in one tree
            'max_depth': trial.suggest_int('max_depth', 4, 12), # Maximum tree depth
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 60), # Minimum number of data needed in a child leaf
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.9, 1.0), # Fraction of features for training each tree
            'random_state': 42, # Keep random state fixed
            'objective': 'tweedie' # Using MAE as internal objective - consider experimenting
            # You could also try 'regression' (MSE) or other regression objectives here
        }

        # Initialize and train the model on the evaluation training set
        model = lgb.LGBMRegressor(**params)

        # Train the model with the custom evaluation metric
        model.fit(X_train_eval, y_train_eval,
                  categorical_feature=eval_categorical_cols,
                  # Use validation set with the custom metric
                  eval_set=[(X_val_eval, y_val_eval)],
                  eval_metric=custom_metric_eval, # Use the custom metric class instance
                  callbacks=[lgb.early_stopping(10, verbose=False)] # Remove the monitor parameter
                 )

        # Make predictions on the validation set
        predictions_val = model.predict(X_val_eval)

        # --- Calculate the metric for Optuna's objective value ---
        # Use the same logic as the custom_metric_eval, calculated on the final predictions
        # This needs to align preds and labels with product_id from the original df_val_eval_full
        df_temp = df_train_fe.loc[df_train_fe.index.intersection(val_eval_indices)].copy()
        df_temp['preds'] = predictions_val # Use predictions from the model trained in this trial
        df_temp['labels'] = y_val_eval # True labels

        por_producto = df_temp.groupby(custom_metric_eval.product_id_col, observed=True).agg(
            {'labels': 'sum', 'preds': 'sum'}
        )

        sum_total_real_global = por_producto['labels'].sum()
        sum_abs_errors_per_product = np.sum(np.abs(por_producto['labels'] - por_producto['preds']))

        if sum_total_real_global == 0:
             objective_value = 0 if sum_abs_errors_per_product == 0 else sum_abs_errors_per_product
        else:
             objective_value = sum_abs_errors_per_product / sum_total_real_global * 100 # Percentage

        # Return the calculated metric for Optuna to minimize
        return objective_value # Optuna minimiza este value

    # --- Run Optimization ---
    lgbm_final = load_model_checkpoint(FINAL_MODEL_CHECKPOINT) # Try loading existing model first

    if lgbm_final is None and not X_train_eval.empty and not X_val_eval.empty:
        print("\nStarting Optuna hyperparameter optimization...")
        # Create a study object and specify the direction of optimization
        study = optuna.create_study(direction='minimize')

        # Run the optimization for a limited number of trials or a fixed duration
        # Adjust n_trials or timeout based on how much time you want to spend
        study.optimize(objective, n_trials=5) # Run 5 trials (adjust as needed)
        # Alternative: study.optimize(objective, timeout=600) # Run for max 600 seconds (10 minutes)

        print("\nOptuna optimization finished.")
        print("Best trial:")
        print(f"  Value: {study.best_trial.value:.4f}")
        print("  Params: ")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")

        # --- Train Final Model with Best Parameters ---
        print(f"\nTraining final model with best parameters on full historical data up to {LAST_HISTORICAL_PERIOD}...")
        best_params = study.best_trial.params

        # Retrain on full X_train, y_train with best params
        if not X_train.empty and not y_train.empty:
            lgbm_final = lgb.LGBMRegressor(**best_params)
            lgbm_final.fit(X_train, y_train, categorical_feature=eval_categorical_cols)
            save_model_checkpoint(lgbm_final, FINAL_MODEL_CHECKPOINT)

    elif lgbm_final is None:
         print("\nSkipping Optuna optimization and final model training due to insufficient evaluation or training data.")

    # --- Generate Feature Importance Plot ---
    if lgbm_final is not None:
        print("\nGenerating feature importance plot...")
        # Get the feature names from the model
        feature_names = lgbm_final.feature_name_
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': lgbm_final.feature_importances_
        })
        importance = importance.sort_values('importance', ascending=False).head(40)

        plt.figure(figsize=(10, 7))
        ax = sns.barplot(x='importance', y='feature', data=importance, palette='viridis')
        plt.title(f'Top 40 Feature Importance (Predicting {PREDICTION_PERIOD})', fontsize=16)
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)

        for p in ax.patches:
            width = p.get_width()
            format_string = '{:1.4f}' if width < 0.01 else '{:1.2f}'
            plt.text(width + 0.005, p.get_y() + p.get_height()/2.,
                    format_string.format(width),
                    ha='left', va='center', fontsize=10)

        plt.tight_layout()
        importance_plot_path = os.path.join(CHECKPOINTS_DIR, f'feature_importance_{PREDICTION_PERIOD}.png')
        plt.savefig(importance_plot_path, bbox_inches='tight')
        print(f"Feature importance plot saved to {importance_plot_path}")

    # --- Make Predictions ---
    prediction_results_df = load_dataframe_checkpoint(PREDICTIONS_DF_CHECKPOINT)

    if prediction_results_df is None and lgbm_final is not None:
        print(f"\nMaking predictions for period {PREDICTION_PERIOD}...")
        predictions_predict = lgbm_final.predict(X_predict)

        # Create results dataframe with IDs and fecha
        prediction_results_df = df_predict_fe_initial[['customer_id', 'product_id', 'fecha', TARGET]].copy()
        prediction_results_df[f'{TARGET}_predicted_{PREDICTION_PERIOD}'] = predictions_predict
        save_dataframe_checkpoint(prediction_results_df, PREDICTIONS_DF_CHECKPOINT)

    # Save predictions to CSV
    if prediction_results_df is not None and not prediction_results_df.empty:
        predictions_output_path = os.path.join(CHECKPOINTS_DIR, f'predictions_{PREDICTION_PERIOD}.csv')
        prediction_results_df.to_csv(predictions_output_path, index=False)
        print(f"\nPredictions saved to {predictions_output_path}")
        print("\nPrediction Examples:")
        print(prediction_results_df.head())

    print("\nModel Training and Prediction Script Finished.")
