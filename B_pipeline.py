#!/usr/bin/env python

import pandas as pd
import os
import numpy as np
import time

# Import functions from the helper script
from A_funciones_pipeline import (
    cargar_y_combinar_datos,
    transformar_periodo,
    generar_combinaciones_por_periodo,
    merge_with_original_data,
    fill_missing_product_info,
    generar_lags_por_combinacion,
    calculate_brand_loyalty,
    add_customer_category_avg_tn,
    calculate_product_moving_avg,
    calculate_tn_percentage_change,
    calculate_months_since_last_purchase,
    calculate_customer_category_count,
    add_macro_event_flag,
    calculate_weighted_tn_sum,
    calculate_demand_growth_rate_diff,
    #calculate_cust_request_tn_anomaly,
)

# Define data paths (assuming script is run from Maestría/20- Laboratorio3/Data/)
SELL_IN_PATH = 'sell-in.txt'
PRODUCTOS_PATH = 'tb_productos.txt'
STOCKS_PATH = 'tb_stocks.txt'

# Define checkpoint paths
CHECKPOINTS_DIR = r'C:\Users\Martin\OneDrive\Maestría\20- Laboratorio3\Data\checkpoints' # Updated to absolute path
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# Intermediate checkpoints (up to the split point)
DF_INITIAL_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, 'df_inicial.pkl')
DF_CON_FECHA_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, 'df_con_fecha.pkl')
DF_MERGED_COMBINATIONS_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, 'df_merged_combinations.pkl')
DF_FILLED_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, 'df_filled.pkl') # <-- Data before leakage FE

# Final output checkpoints (after split and feature engineering)
DF_TRAIN_FINAL_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, 'df_train_final_featured.pkl') # <-- New: Final training data output
DF_PREDICT_FINAL_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, 'df_predict_final_featured.pkl') # <-- New: Final prediction data output

# Feature engineering checkpoints (ordered by execution sequence)
DF_01_BRAND_LOYALTY_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, '01_brand_loyalty.pkl')
DF_02_CUSTOMER_CATEGORY_AVG_TN_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, '02_customer_category_avg_tn.pkl')
DF_03_CUSTOMER_CATEGORY_COUNT_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, '03_customer_category_count.pkl')
DF_04_MACRO_EVENT_FLAG_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, '04_macro_event_flag.pkl')
DF_05_TN_PERCENTAGE_CHANGE_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, '05_tn_percentage_change.pkl')
DF_06_MONTHS_SINCE_LAST_PURCHASE_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, '06_months_since_last_purchase.pkl')
DF_07_PRODUCT_MOVING_AVG_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, '07_product_moving_avg.pkl')
DF_08_WEIGHTED_TN_SUM_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, '08_weighted_tn_sum.pkl')
DF_09_DEMAND_GROWTH_RATE_DIFF_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, '09_demand_growth_rate_diff.pkl')
#DF_10_CUST_REQUEST_TN_ANOMALY_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, '10_cust_request_tn_anomaly.pkl')
DF_11_LAGS_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, '11_lags.pkl')

# Define the target prediction period (format YYYYMM) - This is the period we want to predict
PREDICTION_PERIOD = 202002
# Define the date corresponding to the prediction period
PREDICTION_DATE = pd.to_datetime(str(PREDICTION_PERIOD), format='%Y%m').to_period('M')

# Define the last available historical data period (for training)
LAST_HISTORICAL_PERIOD = 201912
LAST_HISTORICAL_DATE = pd.to_datetime(str(LAST_HISTORICAL_PERIOD), format='%Y%m').to_period('M')

# Parameters for feature engineering functions (Define here as pipeline applies them)
LAG_COLUMNS = ['cust_request_qty', 'cust_request_tn','tn']
NUM_LAGS = 12
# Moving average window (e.g., 3 months)
MOVING_AVG_WINDOW = 3 # Note: The calculate_product_moving_avg function in your helper script hardcoded 3
TARGET = 'tn' # The original target column name before shifting
TARGET_SHIFT = 2 # Shift for the future target (t+2)
FUTURE_TARGET = f'{TARGET}_future' # New column name for the shifted target

# Define list of steps up to the point *before* the time-series split and specific FE
# These steps can be run sequentially and checkpointed as before.
initial_pipeline_steps = [
    (cargar_y_combinar_datos, DF_INITIAL_CHECKPOINT, "Load and combine initial data", False), # False means function doesn't take df
    (transformar_periodo, DF_CON_FECHA_CHECKPOINT, "Transform periodo to date", True), # True means function takes df
    (generar_combinaciones_por_periodo, None, "Generate combinations", True), # Special case: combinations_df is an intermediate, not a final checkpoint for splitting
    (merge_with_original_data, DF_MERGED_COMBINATIONS_CHECKPOINT, "Merge combinations with original data", True), # Takes combinations_df and original_df, needs special handling
    (fill_missing_product_info, DF_FILLED_CHECKPOINT, "Fill missing product info", True) # Takes df and products_df, needs special handling
]

# Define feature engineering steps with their checkpoints (in the same order)
feature_engineering_steps = [
    (calculate_brand_loyalty, DF_01_BRAND_LOYALTY_CHECKPOINT, "Calculate brand loyalty", ()),
    (add_customer_category_avg_tn, DF_02_CUSTOMER_CATEGORY_AVG_TN_CHECKPOINT, "Add customer category average tn", ()),
    (calculate_customer_category_count, DF_03_CUSTOMER_CATEGORY_COUNT_CHECKPOINT, "Calculate customer category count", ()),
    (add_macro_event_flag, DF_04_MACRO_EVENT_FLAG_CHECKPOINT, "Add macro event flag", ()),
    (calculate_tn_percentage_change, DF_05_TN_PERCENTAGE_CHANGE_CHECKPOINT, "Calculate tn percentage change", ()),
    (calculate_months_since_last_purchase, DF_06_MONTHS_SINCE_LAST_PURCHASE_CHECKPOINT, "Calculate months since last purchase", ()),
    (calculate_product_moving_avg, DF_07_PRODUCT_MOVING_AVG_CHECKPOINT, "Calculate product moving average", ()),
    (calculate_weighted_tn_sum, DF_08_WEIGHTED_TN_SUM_CHECKPOINT, "Calculate weighted tn sum", ()),
    (calculate_demand_growth_rate_diff, DF_09_DEMAND_GROWTH_RATE_DIFF_CHECKPOINT, "Calculate demand growth rate diff", ()),
    #(calculate_cust_request_tn_anomaly, DF_10_CUST_REQUEST_TN_ANOMALY_CHECKPOINT, "Calculate cust request tn anomaly", ()), muy pesado y largo, y no suma tanto, lo comento
    (generar_lags_por_combinacion, DF_11_LAGS_CHECKPOINT, "Calculate lags", (LAG_COLUMNS, NUM_LAGS)),
]

# --- Helper function to run initial steps with checkpoints ---
def run_initial_pipeline_steps():
    df_current = None
    start_step_index = 0
    latest_checkpoint_found = False
    combinations_df = None # Initialize combinations_df outside the loop

    print("Checking for latest initial checkpoint...")
    try:
        stop_checkpoint = DF_FILLED_CHECKPOINT
        # Find the index in the list based on the checkpoint path
        stop_index_in_list = next((i for i, (_, cp, _, _) in enumerate(initial_pipeline_steps) if cp == stop_checkpoint), -1)

        if stop_index_in_list == -1:
             raise ValueError(f"Stop checkpoint {stop_checkpoint} not found in initial_pipeline_steps list.")

        # Look for checkpoints in reverse order up to the stop_checkpoint
        for i in range(stop_index_in_list, -1, -1):
             step_func, checkpoint_path, description, takes_df = initial_pipeline_steps[i]
             if checkpoint_path is not None and os.path.exists(checkpoint_path):
                  print(f"Latest initial checkpoint found: {description} from {checkpoint_path}")
                  df_current = pd.read_pickle(checkpoint_path)
                  start_step_index = i + 1 # Start processing from the next step
                  latest_checkpoint_found = True
                  break # Found the latest, no need to check earlier ones

    except Exception as e:
        print(f"Error during initial checkpoint check: {e}")
        print("Starting initial pipeline from the beginning.")
        start_step_index = 0
        df_current = None
        latest_checkpoint_found = False
        combinations_df = None # Reset combinations_df

    if not latest_checkpoint_found:
        print("No initial checkpoints found (up to the intended stop point). Starting from the beginning.")

    print(f"\nRunning initial pipeline steps from index {start_step_index}...")

    # Execute the steps from the determined starting index up to the intended stop point
    products_df = None # Initialize products_df for fill_missing_product_info

    for i in range(start_step_index, len(initial_pipeline_steps)):
        step_func, checkpoint_path, description, takes_df = initial_pipeline_steps[i]
        print(f"Executing initial step {i + 1}: {description}")

        # Determine if this step should be skipped because we loaded from a checkpoint
        skip_step_execution = (i == start_step_index and latest_checkpoint_found and start_step_index > 0)

        # --- Handle specific steps ---

        if step_func == cargar_y_combinar_datos:
            if skip_step_execution:
                 print(f"Skipping execution of step {i + 1} ('{description}') as df_current loaded from checkpoint.")
            else:
                 df_current = step_func()

        elif step_func == transformar_periodo:
             if skip_step_execution:
                 print(f"Skipping execution of step {i + 1} ('{description}') as df_current loaded from checkpoint.")
             else:
                 if df_current is None: raise ValueError("df_current is None before transformar_periodo")
                 df_current = step_func(df_current)

        elif step_func == generar_combinaciones_por_periodo:
             # ALWAYS generate combinations_df when this step is reached in the loop
             # This step doesn't modify df_current, it just produces combinations_df
             if df_current is None: raise ValueError("df_current is None before generar_combinaciones_por_periodo")
             combinations_df = step_func(df_current)
             print("combinations_df generated.")
             # No checkpoint for combinations_df as it's immediately used by the next step
             # We do NOT skip the rest of the loop body after this step, as we need to save the result of the *next* step.

        elif step_func == merge_with_original_data:
             # This step needs combinations_df and the original df_con_fecha (in df_current)
             # combinations_df should have been generated by the previous step (index i-1)
             if skip_step_execution:
                 print(f"Skipping execution of step {i + 1} ('{description}') as df_current loaded from checkpoint.")
                 # If we skipped this step, df_current is the loaded checkpoint data (result *after* this step)
                 # combinations_df is no longer needed for the pipeline flow
                 combinations_df = None # Clean up intermediate variable
             else:
                 # Ensure combinations_df is available when running this step
                 if df_current is None or combinations_df is None:
                      raise ValueError("Missing data for merge_with_original_data: df_current or combinations_df is None.")

                 # Merge combinations_df with the result of previous step (df_con_fecha)
                 df_current = step_func(combinations_df, df_current)
                 # Now df_current holds the merged result
                 combinations_df = None # Clean up intermediate variable


        elif step_func == fill_missing_product_info:
             if skip_step_execution:
                 print(f"Skipping execution of step {i + 1} ('{description}') as df_current loaded from checkpoint.")
             else:
                 # This step needs df_current (merged df) and products_df
                 if df_current is None: raise ValueError("df_current is None before fill_missing_product_info")
                 # Load products_df specifically for this step if needed
                 if products_df is None:
                      try:
                           products_df = pd.read_csv(PRODUCTOS_PATH, delimiter='\t')
                           products_df = products_df.drop_duplicates(subset=['product_id'])
                      except Exception as e:
                           raise RuntimeError(f"Could not load products_df from {PRODUCTOS_PATH}: {e}")

                 df_current = step_func(df_current, products_df)

        elif takes_df: # Standard step that takes the previous df as input
             if skip_step_execution:
                 print(f"Skipping execution of step {i + 1} ('{description}') as df_current loaded from checkpoint.")
             else:
                 if df_current is None:
                     raise ValueError(f"DataFrame is None before executing step {i+1} ('{description}').")
                 df_current = step_func(df_current)
        else: # Standard step that doesn't take df (should only be the first step)
             # This case should ideally not be reached for any step after the first one.
             if i == 0:
                  # Assuming the very first step doesn't take df and produces the initial df
                  if skip_step_execution:
                      print(f"Skipping execution of step {i + 1} ('{description}') as df_current loaded from checkpoint.")
                  else:
                       df_current = step_func()
             else:
                 # This indicates a misconfiguration in initial_pipeline_steps regarding 'takes_df'.
                 raise ValueError(f"Unexpected step type or position: {description} at index {i}. 'takes_df' might be incorrectly set.")


        # Save checkpoint after executing the step, if a path is defined
        # Only save if df_current is not None after the step
        if checkpoint_path is not None and df_current is not None:
             print(f"Saving intermediate checkpoint to {checkpoint_path}")
             df_current.to_pickle(checkpoint_path)

        # Stop after processing the intended stop point (DF_FILLED_CHECKPOINT)
        if checkpoint_path == stop_checkpoint:
             print(f"Initial pipeline steps completed up to {stop_checkpoint}.")
             break # Exit the loop after saving the stop checkpoint


    # Final check and load if loop finished before reaching stop_checkpoint or df_current is None
    if df_current is None or (stop_checkpoint is not None and not os.path.exists(stop_checkpoint)):
        if df_current is None and stop_checkpoint is not None and os.path.exists(stop_checkpoint):
             print(f"Loading final initial checkpoint {stop_checkpoint} after loop completion.")
             df_current = pd.read_pickle(stop_checkpoint)
        elif df_current is None:
             raise RuntimeError("Initial pipeline failed to produce a DataFrame up to the stop point.")
        # Add a check if stop_checkpoint was reached but not saved for some reason
        elif stop_checkpoint is not None and not os.path.exists(stop_checkpoint):
             print(f"Warning: Loop completed but stop checkpoint {stop_checkpoint} was not saved. Returning df_current assuming it's the correct state.")


    return df_current # Return the DataFrame at the state of DF_FILLED_CHECKPOINT

def run_feature_engineering_steps(df):
    """
    Run feature engineering steps with checkpointing.
    
    Args:
        df (pd.DataFrame): Input DataFrame to apply feature engineering to
        
    Returns:
        pd.DataFrame: DataFrame with all features engineered
    """
    df_current = df.copy()
    start_step_index = 0
    latest_checkpoint_found = False
    
    print("Checking for latest feature engineering checkpoint...")
    
    # Look for checkpoints in reverse order
    for i in range(len(feature_engineering_steps) - 1, -1, -1):
        step_func, checkpoint_path, description, _ = feature_engineering_steps[i]
        if os.path.exists(checkpoint_path):
            print(f"Latest feature engineering checkpoint found: {description} from {checkpoint_path}")
            df_current = pd.read_pickle(checkpoint_path)
            start_step_index = i + 1
            latest_checkpoint_found = True
            break
    
    if not latest_checkpoint_found:
        print("No feature engineering checkpoints found. Starting from the beginning.")
    
    print(f"\nRunning feature engineering steps from index {start_step_index}...")
    
    # Execute the steps from the determined starting index
    for i in range(start_step_index, len(feature_engineering_steps)):
        step_func, checkpoint_path, description, args = feature_engineering_steps[i]
        print(f"Executing feature engineering step {i + 1}: {description}")
        
        try:
            start_time = time.time()
            # Pass the dataframe and any additional arguments
            df_current = step_func(df_current, *args)
            print(f"{description} calculated in {time.time() - start_time:.2f} seconds.")
            
            # Save checkpoint after successful execution
            print(f"Saving checkpoint to {checkpoint_path}")
            df_current.to_pickle(checkpoint_path)
            
        except Exception as e:
            print(f"Error during {description}: {e}")
            print(f"Pipeline stopped at {description}. You can resume from the last successful checkpoint.")
            raise
    
    return df_current

# --- Main Execution Flow ---
if __name__ == "__main__":
    print("Starting Data Pipeline Script with Time-Based Split and Leakage-Free FE")

    # --- Step 1: Run initial processing steps up to the point before leakage FE ---
    # This function handles loading from checkpoints or running from scratch up to df_filled.pkl
    df_pre_fe = run_initial_pipeline_steps()

    if df_pre_fe is None:
        raise RuntimeError("Initial pipeline steps failed to produce the data needed for feature engineering.")

    # Ensure fecha is datetime for splitting and shifting
    if not pd.api.types.is_datetime64_any_dtype(df_pre_fe['fecha']):
        df_pre_fe['fecha'] = pd.to_datetime(df_pre_fe['fecha'])

    # --- Step 2: Separate Train and Predict Data (Initial Split) ---
    print(f"\nSeparating raw data for Training (up to {LAST_HISTORICAL_PERIOD}) and Prediction ({PREDICTION_PERIOD})")

    # Historical data: All data up to the last available period (201912)
    df_historical_raw = df_pre_fe[df_pre_fe['fecha'] <= pd.to_datetime(str(LAST_HISTORICAL_PERIOD), format='%Y%m')].copy()

    # Prediction data: Generate combinations for the target prediction period (202002)
    # We need unique customer_id and product_id from the *entire* historical data for prediction rows
    unique_combinations = df_historical_raw[['customer_id', 'product_id']].drop_duplicates().copy()

    # Create the prediction dataframe rows with the target prediction date
    df_predict_raw = unique_combinations.copy()
    df_predict_raw['fecha'] = pd.to_datetime(str(PREDICTION_PERIOD), format='%Y%m')
    df_predict_raw['periodo'] = PREDICTION_PERIOD # Add periodo column
    # The 'tn' column will be NaN for these rows, which is correct as it's unknown.

    print(f"Historical raw data shape (up to {LAST_HISTORICAL_PERIOD}): {df_historical_raw.shape}")
    print(f"Predict raw data shape (for {PREDICTION_PERIOD}): {df_predict_raw.shape}")

    # --- Step 3: Calculate Target Variable on Historical Data Only ---
    # The target for a row with date 't' will be the 'tn' value at date 't + TARGET_SHIFT'
    # We calculate this ONLY on historical data first to determine the actual training set.
    print(f"\nCalculating target '{FUTURE_TARGET}' on historical data...")

    # Sort historical data before shifting
    df_historical_raw = df_historical_raw.sort_values(by=['customer_id', 'product_id', 'fecha'])

    # Calculate the future target for each group in historical data
    df_historical_raw[FUTURE_TARGET] = df_historical_raw.groupby(['customer_id', 'product_id'])[TARGET].shift(-TARGET_SHIFT)

    # --- Step 4: Define Final Training Set (Historical data where FUTURE_TARGET is NOT NaN) ---
    # This excludes the last TARGET_SHIFT periods from the training set because their future target is unknown.
    df_train_final_pre_fe = df_historical_raw[df_historical_raw[FUTURE_TARGET].notna()].copy()
    
    # Also keep the full historical data (including the last two periods) for feature calculation later
    df_historical_full_pre_fe = df_historical_raw.copy()
    # Drop the FUTURE_TARGET column from the full historical data before FE, it's only for the final train set
    df_historical_full_pre_fe = df_historical_full_pre_fe.drop(columns=[FUTURE_TARGET])


    print(f"Initial Train Final data shape (historical data excluding last {TARGET_SHIFT} periods): {df_train_final_pre_fe.shape}")

    # --- Step 5: Prepare Combined Data for Feature Engineering ---
    # Combine the full historical data (including the last {TARGET_SHIFT} periods) and prediction data
    # Feature engineering needs the data leading up to the prediction period.
    print("\nCombining full historical and prediction raw data for feature engineering...")
    df_combined_for_fe = pd.concat([df_historical_full_pre_fe, df_predict_raw], ignore_index=True)

    # Ensure combined data is sorted for time-dependent features
    df_combined_for_fe = df_combined_for_fe.sort_values(by=['customer_id', 'product_id', 'fecha'])

    print(f"Combined data shape for Feature Engineering: {df_combined_for_fe.shape}")

    # --- Step 6: Apply Feature Engineering on Combined Data ---
    print("\nApplying feature engineering on combined data...")
    df_combined_fe = run_feature_engineering_steps(df_combined_for_fe)

    # --- Step 7: Separate the Featured Data Back into Train and Predict ---
    # Now we split the featured data.
    # Featured Train Data: Match the periods and combinations present in df_train_final_pre_fe
    # We need to merge the features from df_combined_fe back to df_train_final_pre_fe

    print("\nSeparating featured data back into Train and Predict sets...")

    # Ensure fecha is Period[M] for easier merging if necessary, or keep as datetime and merge on datetime
    # Let's stick to datetime as it's used throughout
    df_combined_fe['fecha'] = pd.to_datetime(df_combined_fe['fecha'])
    df_train_final_pre_fe['fecha'] = pd.to_datetime(df_train_final_pre_fe['fecha'])


    # Merge features from df_combined_fe onto the structure of df_train_final_pre_fe
    # Keep all columns from df_combined_fe except the original TARGET and FUTURE_TARGET (which is NaN for most rows here)
    # We will use the FUTURE_TARGET calculated in Step 3.
    cols_to_merge = [col for col in df_combined_fe.columns if col not in [TARGET, FUTURE_TARGET]] # Excluir TARGET y FUTURE_TARGET

    print(f"Columns in df_train_final_pre_fe before merge: {df_train_final_pre_fe.columns.tolist()}")
    print(f"Columns in df_combined_fe before merge: {df_combined_fe.columns.tolist()}")
    print(f"Columns to merge from df_combined_fe: {cols_to_merge}")
    print(f"On columns for merge: {['customer_id', 'product_id', 'fecha']}")


    df_train_final = pd.merge(
        df_train_final_pre_fe[['customer_id', 'product_id', 'fecha', TARGET, FUTURE_TARGET]], # Incluir TARGET y FUTURE_TARGET del df_train_final_pre_fe
        df_combined_fe[cols_to_merge], # Mergear el resto de columnas de características (excluyendo TARGET y FUTURE_TARGET)
        on=['customer_id', 'product_id', 'fecha'],
        how='left' # Left merge para mantener solo las filas definidas en df_train_final_pre_fe
    )


    # Featured Predict Data: Rows where 'fecha' is the PREDICTION_DATE (202002)
    df_predict_final = df_combined_fe[df_combined_fe['fecha'] == pd.to_datetime(str(PREDICTION_PERIOD), format='%Y%m')].copy()

    # Drop FUTURE_TARGET from the prediction set as it's the column we need to predict
    if FUTURE_TARGET in df_predict_final.columns:
         df_predict_final = df_predict_final.drop(columns=[FUTURE_TARGET])


    print(f"Train final data shape (after feature engineering): {df_train_final.shape}")
    print(f"Predict final data shape (for {PREDICTION_PERIOD}): {df_predict_final.shape}")

    if df_train_final.empty:
        print("Critical Warning: Final training data is empty. Cannot train a model.")
    if df_predict_final.empty:
        print("Warning: Final prediction data for target period is empty.")

    # --- Step 8: Convert categorical columns to 'category' dtype ---
    # Identify potential categorical columns *after* FE
    # Check dtypes - object or category are likely candidates
    # Exclude explicit IDs and columns that are now numerical due to FE (lags, etc.)
    # Perform this identification on the training set as it has more data, but apply to both
    # Include 'cliente_categoria' back if it's categorical and should be used as a feature
    # Exclude the target(s) from this conversion list
    cols_to_exclude_from_cat = ['customer_id', 'product_id', 'periodo', 'fecha', TARGET, FUTURE_TARGET] + [col for col in df_train_final.columns if df_train_final[col].dtype in ['int64', 'float64']] # Also exclude already numeric cols

    potential_categorical_cols = [col for col in df_train_final.columns if df_train_final[col].dtype in ['object', 'category'] and col not in cols_to_exclude_from_cat]

    print(f"\nConverting columns to 'category' dtype for LGBM: {potential_categorical_cols}")

    for col in potential_categorical_cols:
        if col in df_train_final.columns:
            df_train_final[col] = df_train_final[col].astype('category')
        else:
            print(f"Warning: Categorical column '{col}' not found in train final data. Skipping.")

        if col in df_predict_final.columns:
             # Ensure categories are consistent between train and predict
             if col in df_train_final.columns:
                  # Use categories from the training data to ensure consistency
                  df_predict_final[col] = df_predict_final[col].astype('category').cat.set_categories(df_train_final[col].cat.categories)
             else: # Should not happen if identified from train_final
                 pass # Added 'pass' here to fix the syntax error
        else: # This 'else' block also needs a pass or code if reached
             print(f"Warning: Categorical column '{col}' not found in predict final data. Skipping.") # This print statement is the body


    # --- Step 8.1: Add date features (month, year, day, dayofweek) ---
    # Ensure fecha is datetime before extracting features
    if not pd.api.types.is_datetime64_any_dtype(df_train_final['fecha']):
         df_train_final['fecha'] = pd.to_datetime(df_train_final['fecha'])
    if not pd.api.types.is_datetime64_any_dtype(df_predict_final['fecha']):
         df_predict_final['fecha'] = pd.to_datetime(df_predict_final['fecha'])


    print("\nAdding date features...")
    df_train_final['month'] = df_train_final['fecha'].dt.month
    df_train_final['year'] = df_train_final['fecha'].dt.year
    # These might not be necessary or useful for monthly data, but adding them as before
    df_train_final['day'] = df_train_final['fecha'].dt.day # This will likely be 1 for all rows
    df_train_final['dayofweek'] = df_train_final['fecha'].dt.dayofweek # This will vary based on the 1st of the month

    df_predict_final['month'] = df_predict_final['fecha'].dt.month
    df_predict_final['year'] = df_predict_final['fecha'].dt.year
    df_predict_final['day'] = df_predict_final['fecha'].dt.day # Likely 1
    df_predict_final['dayofweek'] = df_predict_final['fecha'].dt.dayofweek # Varies

    # --- Step 8.2: Ensure ID columns are category type (moved from train_model.py) ---
    print("\nEnsuring ID columns are category type...")
    id_cols_to_cat = ['customer_id', 'product_id', 'periodo']
    for col in id_cols_to_cat:
        if col in df_train_final.columns:
            df_train_final[col] = df_train_final[col].astype('category')
        else:
            print(f"Warning: ID column '{col}' not found in train final data for category conversion. Skipping.")

        if col in df_predict_final.columns:
             # Ensure categories are consistent between train and predict
             if col in df_train_final.columns:
                  df_predict_final[col] = df_predict_final[col].astype('category').cat.set_categories(df_train_final[col].cat.categories)
             else: # Should not happen if identified from train_final
                  df_predict_final[col] = df_predict_final[col].astype('category')
        else:
             print(f"Warning: ID column '{col}' not found in predict final data for category conversion. Skipping.")


    # --- Step 9: Handle missing values before saving ---
    # Apply handling missing values to the final train and predict sets
    # Note: FUTURE_TARGET NaNs were already handled by filtering the training set.
    # We still need to fill NaNs in features in both train and predict sets.
    def handle_missing_values_final(df):
        # Identify numeric columns (excluding the target in train)
        num_cols_to_fill = [col for col in df.select_dtypes(include=np.number).columns if col != FUTURE_TARGET]
        
        # Identify categorical columns
        cat_cols = df.select_dtypes(include='category').columns

        # Fill numeric NaNs with 0 (or another strategy if preferred)
        # Exclude specific columns you want to leave as NaN for LGBM, if any.
        # Based on previous discussion, we might want to leave lags/moving averages as NaN
        cols_to_leave_as_nan = [col for col in df.columns if 'lag_' in col or 'moving_avg' in col]
        num_cols_for_zero_fill = [col for col in num_cols_to_fill if col not in cols_to_leave_as_nan]

        print(f"Filling numeric NaNs with 0 (excluding lags/moving_avg): {num_cols_for_zero_fill}")
        df[num_cols_for_zero_fill] = df[num_cols_for_zero_fill].fillna(0)

        # Fill categorical NaNs with 'missing' category
        print(f"Filling categorical NaNs with 'missing': {cat_cols.tolist()}")
        for col in cat_cols:
             if 'missing' not in df[col].cat.categories:
                 df[col] = df[col].cat.add_categories('missing')
             df[col] = df[col].fillna('missing')


        return df

    # Apply handling missing values *after* final splitting
    df_train_final = handle_missing_values_final(df_train_final)
    df_predict_final = handle_missing_values_final(df_predict_final) # No FUTURE_TARGET in predict_final at this point


    # --- Step 10: Save final processed DataFrames ---
    print(f"\nSaving final train data to {DF_TRAIN_FINAL_CHECKPOINT}")
    df_train_final.to_pickle(DF_TRAIN_FINAL_CHECKPOINT)

    print(f"Saving final prediction data to {DF_PREDICT_FINAL_CHECKPOINT}")
    df_predict_final.to_pickle(DF_PREDICT_FINAL_CHECKPOINT)

    print("\nPipeline Execution Completed.")
    print(f"Final Train DataFrame shape: {df_train_final.shape}")
    print(f"Final Predict DataFrame shape: {df_predict_final.shape}")
    print("\nColumns in final Train DataFrame:")
    print(df_train_final.columns.tolist())
    print("\nColumns in final Predict DataFrame:")
    print(df_predict_final.columns.tolist())
    
    