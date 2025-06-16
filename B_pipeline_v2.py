#!/usr/bin/env python

import pandas as pd
import os
import numpy as np
import time
import threading

def save_checkpoint_async(df, path):
    def save():
        df.to_pickle(path)
    thread = threading.Thread(target=save)
    thread.start()
    return thread

# Archivo	Propósito
# df_inicial.pkl	Datos combinados iniciales
# df_con_fecha.pkl	Datos con columna de fecha
# df_merged_combinations.pkl	Combinaciones mergeadas con datos originales
# df_filled.pkl	Datos con info de producto completada
# 01_brand_loyalty.pkl	Feature: lealtad de marca
# 02_customer_category_avg_tn.pkl	Feature: promedio tn por cliente/categoría
# 03_customer_category_count.pkl	Feature: conteo de categorías por cliente
# 04_macro_event_flag.pkl	Feature: bandera de evento macroeconómico
# 05_tn_percentage_change.pkl	Feature: cambio porcentual de tn
# 06_months_since_last_purchase.pkl	Feature: meses desde última compra
# 07_product_moving_avg.pkl	Feature: promedio móvil de tn
# 08_weighted_tn_sum.pkl	Feature: suma ponderada de tn
# 09_demand_growth_rate_diff.pkl	Feature: diff. tasa de crecimiento de demanda
# 10_total_tn_per_product.pkl	Feature: suma global de tn por producto
# 11_lags.pkl	Features: columnas de lags
# df_train_final_featured.pkl	DataFrame final de entrenamiento
# df_predict_final_featured.pkl	DataFrame final de predicción

GCS_BUCKET_PATH = '/home/chidiakmartin/gcs-bucket'

SELL_IN_PATH = os.path.join(GCS_BUCKET_PATH, 'sell-in.txt')
PRODUCTOS_PATH = os.path.join(GCS_BUCKET_PATH, 'tb_productos.txt')
STOCKS_PATH = os.path.join(GCS_BUCKET_PATH, 'tb_stocks.txt')  
EVENTOS_PATH = os.path.join(GCS_BUCKET_PATH, 'eventos_macro_arg_2017_2019.txt')  
CHECKPOINTS_DIR = os.path.join(GCS_BUCKET_PATH, 'checkpoints2')

os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# Checkpoints
DF_INITIAL_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, 'df_inicial.pkl')
DF_CON_FECHA_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, 'df_con_fecha.pkl')
DF_MERGED_COMBINATIONS_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, 'df_merged_combinations.pkl')
DF_FILLED_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, 'df_filled.pkl')
DF_TRAIN_FINAL_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, 'df_train_final_featured.pkl')
DF_PREDICT_FINAL_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, 'df_predict_final_featured.pkl')
DF_01_BRAND_LOYALTY_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, '01_brand_loyalty.pkl')
DF_02_CUSTOMER_CATEGORY_AVG_TN_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, '02_customer_category_avg_tn.pkl')
DF_03_CUSTOMER_CATEGORY_COUNT_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, '03_customer_category_count.pkl')
DF_04_MACRO_EVENT_FLAG_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, '04_macro_event_flag.pkl')
DF_05_TN_PERCENTAGE_CHANGE_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, '05_tn_percentage_change.pkl')
DF_06_MONTHS_SINCE_LAST_PURCHASE_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, '06_months_since_last_purchase.pkl')
DF_07_PRODUCT_MOVING_AVG_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, '07_product_moving_avg.pkl')
DF_08_WEIGHTED_TN_SUM_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, '08_weighted_tn_sum.pkl')
DF_09_DEMAND_GROWTH_RATE_DIFF_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, '09_demand_growth_rate_diff.pkl')
DF_10_TOTAL_TN_PER_PRODUCT_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, '10_total_tn_per_product.pkl')
DF_11_LAGS_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, '11_lags.pkl')
DF_12_ADD_ROLLING_STATISTICS_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, '12_add_rolling_statistics_features.pkl')
DF_13_ADD_EXPONENTIAL_MOVING_AVERAGE_FEATURES_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, '13_add_exponential_moving_average_features.pkl')
DF_14_ADD_TREND_FEATURES_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, '14_add_trend_features.pkl')
DF_15_ADD_DIFFERENCE_FEATURES_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, '15_add_difference_features.pkl')
DF_16_ADD_TOTAL_CATEGORY_SALES_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, '16_add_total_category_sales.pkl')
DF_17_ADD_CUSTOMER_PRODUCT_TOTAL_WEIGHTS_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, '17_add_customer_product_total_weights.pkl')
DF_18_ADD_INTERACTION_FEATURES_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, '18_add_interaction_features.pkl')


# Path for the external list of product IDs
LISTADO_IDS_PATH = os.path.join(GCS_BUCKET_PATH, 'ListadoIDS.txt')

PREDICTION_PERIOD = 202002
PREDICTION_DATE = pd.to_datetime(str(PREDICTION_PERIOD), format='%Y%m').to_period('M')
LAST_HISTORICAL_PERIOD = 201912
LAST_HISTORICAL_DATE = pd.to_datetime(str(LAST_HISTORICAL_PERIOD), format='%Y%m').to_period('M')
LAG_COLUMNS = ['cust_request_qty', 'cust_request_tn','tn']
NUM_LAGS = 12
TARGET = 'tn'
TARGET_SHIFT = 2
FUTURE_TARGET = f'{TARGET}_future'


from A_funciones_pipeline_v2 import (
    cargar_y_combinar_datos,
    optimize_dtypes,
    transformar_periodo,
    generar_combinaciones_por_periodo,
    merge_with_original_data,
    fill_missing_product_info,
    calculate_brand_loyalty,
    add_customer_category_avg_tn,
    calculate_product_moving_avg,
    add_macro_event_flag,
    calculate_tn_percentage_change,
    calculate_months_since_last_purchase,
    calculate_customer_category_count,
    calculate_weighted_tn_sum,
    calculate_demand_growth_rate_diff,
    add_total_tn_per_product,
    generar_lags_por_combinacion,
    add_rolling_statistics_features,
    add_exponential_moving_average_features,
    add_trend_features,
    add_difference_features,
    add_total_category_sales,
    add_customer_product_total_weights,
    add_interaction_features,

)

# --- Feature engineering steps ---
feature_engineering_steps_v2 = [
    {
        "func": calculate_brand_loyalty,
        "checkpoint": DF_01_BRAND_LOYALTY_CHECKPOINT,
        "description": "Calculate brand loyalty",
        "params": {}
    },
    {
        "func": add_customer_category_avg_tn,
        "checkpoint": DF_02_CUSTOMER_CATEGORY_AVG_TN_CHECKPOINT,
        "description": "Add customer category average tn",
        "params": {}
    },
    {
        "func": calculate_customer_category_count,
        "checkpoint": DF_03_CUSTOMER_CATEGORY_COUNT_CHECKPOINT,
        "description": "Calculate customer category count",
        "params": {}
    },
    {
        "func": add_macro_event_flag,
        "checkpoint": DF_04_MACRO_EVENT_FLAG_CHECKPOINT,
        "description": "Add macro event flag",
        "params": {"event_file_path": EVENTOS_PATH}
    },
    {
        "func": calculate_tn_percentage_change,
        "checkpoint": DF_05_TN_PERCENTAGE_CHANGE_CHECKPOINT,
        "description": "Calculate tn percentage change",
        "params": {}
    },
    {
        "func": calculate_months_since_last_purchase,
        "checkpoint": DF_06_MONTHS_SINCE_LAST_PURCHASE_CHECKPOINT,
        "description": "Calculate months since last purchase",
        "params": {}
    },
    {
        "func": calculate_product_moving_avg,
        "checkpoint": DF_07_PRODUCT_MOVING_AVG_CHECKPOINT,
        "description": "Calculate product moving average",
        "params": {}
    },
    {
        "func": calculate_weighted_tn_sum,
        "checkpoint": DF_08_WEIGHTED_TN_SUM_CHECKPOINT,
        "description": "Calculate weighted tn sum",
        "params": {"window_size": 3}
    },
    {
        "func": calculate_demand_growth_rate_diff,
        "checkpoint": DF_09_DEMAND_GROWTH_RATE_DIFF_CHECKPOINT,
        "description": "Calculate demand growth rate diff",
        "params": {}
    },
    {
        "func": add_total_tn_per_product,
        "checkpoint": DF_10_TOTAL_TN_PER_PRODUCT_CHECKPOINT,
        "description": "Add total tn per product",
        "params": {}
    },
    {
        "func": generar_lags_por_combinacion,
        "checkpoint": DF_11_LAGS_CHECKPOINT,
        "description": "Calculate lags",
        "params": {"columnas_para_lag": LAG_COLUMNS, "num_lags": NUM_LAGS}
    },
    {
        "func": add_rolling_statistics_features,
        "checkpoint": DF_12_ADD_ROLLING_STATISTICS_CHECKPOINT,
        "description": "Add rolling statistics features",
        "params": {}
    },
    {
        "func": add_exponential_moving_average_features,
        "checkpoint": DF_13_ADD_EXPONENTIAL_MOVING_AVERAGE_FEATURES_CHECKPOINT,
        "description": "Add exponential moving average features",
        "params": {}
    },
    # {
    #     "func": add_trend_features,
    #     "checkpoint": DF_14_ADD_TREND_FEATURES_CHECKPOINT,
    #     "description": "Add trend features",
    #     "params": {}
    # },
    {
        "func": add_difference_features,
        "checkpoint": DF_15_ADD_DIFFERENCE_FEATURES_CHECKPOINT,
        "description": "Add difference features",
        "params": {}
    },
    {
        "func": add_total_category_sales,
        "checkpoint": DF_16_ADD_TOTAL_CATEGORY_SALES_CHECKPOINT,
        "description": "Add total category sales features",
        "params": {}
    },
    {
        "func": add_customer_product_total_weights,
        "checkpoint": DF_17_ADD_CUSTOMER_PRODUCT_TOTAL_WEIGHTS_CHECKPOINT,
        "description": "Add customer and product total weights",
        "params": {}
    },
    {
        "func": add_interaction_features,
        "checkpoint": DF_18_ADD_INTERACTION_FEATURES_CHECKPOINT,
        "description": "Add interaction features (e.g., product of tn and sku_size)",
        "params": {}
    },
]

def run_feature_engineering_steps_v2(df, steps):
    df_current = df.copy()
    start_step_index = 0
    latest_checkpoint_found = False
    checkpoint_threads = [] 
    for i in range(len(steps) - 1, -1, -1):
        step = steps[i]
        if os.path.exists(step["checkpoint"]):
            print(f"Latest checkpoint found: {step['description']} from {step['checkpoint']}")
            df_current = pd.read_pickle(step["checkpoint"])
            start_step_index = i + 1
            latest_checkpoint_found = True
            break
    if not latest_checkpoint_found:
        print("No checkpoints found. Starting from the beginning.")
    for i in range(start_step_index, len(steps)):
        step = steps[i]
        print(f"Executing step {i + 1}: {step['description']}")
        try:
            params = step.get("params", {})
            start_time = time.time()
            df_current = step["func"](df_current, **params)
            elapsed = time.time() - start_time
            print(f"{step['description']} completed in {elapsed:.2f} seconds.")
            print(f"Saving checkpoint to {step['checkpoint']}")
            checkpoint_threads.append(save_checkpoint_async(df_current, step["checkpoint"]))
        except Exception as e:
            print(f"Error during {step['description']}: {e}")
            raise
    for t in checkpoint_threads:
        t.join()
    return df_current

# --- Initial pipeline steps (igual que antes, pero usando **params) ---
def run_initial_pipeline_steps_v2():
    df_current = None
    start_step_index = 0
    latest_checkpoint_found = False
    combinations_df = None
    products_df = None

    initial_pipeline_steps = [
        (cargar_y_combinar_datos, DF_INITIAL_CHECKPOINT, "Load and combine initial data", False),
        (transformar_periodo, DF_CON_FECHA_CHECKPOINT, "Transform periodo to date", True),
        (generar_combinaciones_por_periodo, None, "Generate combinations", True),
        (merge_with_original_data, DF_MERGED_COMBINATIONS_CHECKPOINT, "Merge combinations with original data", True),
        (fill_missing_product_info, DF_FILLED_CHECKPOINT, "Fill missing product info", True)
    ]

    stop_checkpoint = DF_FILLED_CHECKPOINT
    stop_index_in_list = next((i for i, (_, cp, _, _) in enumerate(initial_pipeline_steps) if cp == stop_checkpoint), -1)
    for i in range(stop_index_in_list, -1, -1):
        step_func, checkpoint_path, description, takes_df = initial_pipeline_steps[i]
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            print(f"Latest initial checkpoint found: {description} from {checkpoint_path}")
            df_current = pd.read_pickle(checkpoint_path)
            start_step_index = i + 1
            latest_checkpoint_found = True
            break
    if not latest_checkpoint_found:
        print("No initial checkpoints found (up to the intended stop point). Starting from the beginning.")
    for i in range(start_step_index, len(initial_pipeline_steps)):
        step_func, checkpoint_path, description, takes_df = initial_pipeline_steps[i]
        print(f"Executing initial step {i + 1}: {description}")
        skip_step_execution = (i == start_step_index and latest_checkpoint_found and start_step_index > 0)
        start_time = time.time()
        if step_func == cargar_y_combinar_datos:
            if not skip_step_execution:
                df_current = step_func(SELL_IN_PATH, PRODUCTOS_PATH, STOCKS_PATH)
                df_current = optimize_dtypes(df_current)
        elif step_func == transformar_periodo:
            if not skip_step_execution:
                df_current = step_func(df_current)
        elif step_func == generar_combinaciones_por_periodo:
            if df_current is None: raise ValueError("df_current is None before generar_combinaciones_por_periodo")
            combinations_df = step_func(df_current)
        elif step_func == merge_with_original_data:
            if not skip_step_execution:
                if df_current is None or combinations_df is None:
                    raise ValueError("Missing data for merge_with_original_data: df_current or combinations_df is None.")
                df_current = step_func(combinations_df, df_current)
                combinations_df = None
        elif step_func == fill_missing_product_info:
            if not skip_step_execution:
                if df_current is None: raise ValueError("df_current is None before fill_missing_product_info")
                if products_df is None:
                    products_df = pd.read_csv(PRODUCTOS_PATH, delimiter='\t')
                    products_df = products_df.drop_duplicates(subset=['product_id'])
                df_current = step_func(df_current, products_df=products_df)
        elapsed = time.time() - start_time
        print(f"{description} completed in {elapsed:.2f} seconds.")
        if checkpoint_path is not None and df_current is not None:
            print(f"Saving intermediate checkpoint to {checkpoint_path}")
            df_current.to_pickle(checkpoint_path)
        if checkpoint_path == stop_checkpoint:
            print(f"Initial pipeline steps completed up to {stop_checkpoint}.")
            break
    if df_current is None or (stop_checkpoint is not None and not os.path.exists(stop_checkpoint)):
        if df_current is None and stop_checkpoint is not None and os.path.exists(stop_checkpoint):
            print(f"Loading final initial checkpoint {stop_checkpoint} after loop completion.")
            df_current = pd.read_pickle(stop_checkpoint)
        elif df_current is None:
            raise RuntimeError("Initial pipeline failed to produce a DataFrame up to the stop point.")
    return df_current

if __name__ == "__main__":
    print("Starting Data Pipeline Script with Time-Based Split and Leakage-Free FE")
    df_pre_fe = run_initial_pipeline_steps_v2()
    if df_pre_fe is None:
        raise RuntimeError("Initial pipeline steps failed to produce the data needed for feature engineering.")
    if not pd.api.types.is_datetime64_any_dtype(df_pre_fe['fecha']):
        df_pre_fe['fecha'] = pd.to_datetime(df_pre_fe['fecha'])

    print(f"\nSeparating raw data for Training (up to {LAST_HISTORICAL_PERIOD}) and Prediction ({PREDICTION_PERIOD})")
    df_historical_raw = df_pre_fe[df_pre_fe['fecha'] <= pd.to_datetime(str(LAST_HISTORICAL_PERIOD), format='%Y%m')].copy()

    # 1. Obtener las combinaciones únicas customer_id-product_id históricas
    original_unique_customer_product_pairs = df_historical_raw[['customer_id', 'product_id']].drop_duplicates().copy()
    
    # 2. Cargar el listado externo de product_ids
    truly_new_product_ids = np.array([])
    try:
        external_product_ids_df = pd.read_csv(LISTADO_IDS_PATH, delimiter='\t')
        if 'product_id' in external_product_ids_df.columns:
            new_external_product_ids = external_product_ids_df['product_id'].unique()
            # Filtrar los product_ids que ya están en los datos históricos
            historical_product_ids_only = original_unique_customer_product_pairs['product_id'].unique()
            truly_new_product_ids = np.setdiff1d(new_external_product_ids, historical_product_ids_only)
            if len(truly_new_product_ids) > 0:
                print(f"Found {len(truly_new_product_ids)} truly new product_ids in {LISTADO_IDS_PATH} not present historically.")
            else:
                print(f"No truly new product_ids found in {LISTADO_IDS_PATH} that were not already in historical data.")
        else:
            print(f"Warning: 'product_id' column not found in {LISTADO_IDS_PATH}. No external product IDs added.")
    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        print(f"Warning: Could not load or parse {LISTADO_IDS_PATH} ({e}). No external product IDs added.")
    except Exception as e:
        print(f"An unexpected error occurred while processing {LISTADO_IDS_PATH}: {e}. No external product IDs added.")

    # 3. Si hay nuevos product_ids, generar combinaciones con todos los customer_ids históricos
    if len(truly_new_product_ids) > 0:
        all_historical_customer_ids = df_historical_raw['customer_id'].unique()
        
        # Crear un DataFrame de combinaciones nuevas
        # pd.MultiIndex.from_product es eficiente para crear el producto cartesiano
        new_combinations_index = pd.MultiIndex.from_product(
            [all_historical_customer_ids, truly_new_product_ids],
            names=['customer_id', 'product_id']
        )
        new_combinations_df = new_combinations_index.to_frame(index=False)
        
        # 4. Concatenar las combinaciones históricas originales con las nuevas
        final_combinations_for_prediction = pd.concat([original_unique_customer_product_pairs, new_combinations_df]).drop_duplicates().reset_index(drop=True)
        print(f"Total combinations for prediction (historical + new external products): {final_combinations_for_prediction.shape[0]}")
    else:
        final_combinations_for_prediction = original_unique_customer_product_pairs.copy()
        print(f"Total combinations for prediction (historical only): {final_combinations_for_prediction.shape[0]}")


    # Crear df_predict_raw usando las combinaciones finales
    df_predict_raw = final_combinations_for_prediction.copy()
    df_predict_raw['fecha'] = pd.to_datetime(str(PREDICTION_PERIOD), format='%Y%m')
    df_predict_raw['periodo'] = PREDICTION_PERIOD

    print(f"Historical raw data shape (up to {LAST_HISTORICAL_PERIOD}): {df_historical_raw.shape}")
    print(f"Predict raw data shape (for {PREDICTION_PERIOD}): {df_predict_raw.shape}")
    
    print(f"\nCalculating target '{FUTURE_TARGET}' on historical data...")
    df_historical_raw = df_historical_raw.sort_values(by=['customer_id', 'product_id', 'fecha'])
    df_historical_raw[FUTURE_TARGET] = df_historical_raw.groupby(['customer_id', 'product_id'])[TARGET].shift(-TARGET_SHIFT)
    df_train_final_pre_fe = df_historical_raw[df_historical_raw[FUTURE_TARGET].notna()].copy()
    print("Shape de df_train_final_pre_fe:", df_train_final_pre_fe.shape)
    df_historical_full_pre_fe = df_historical_raw.copy()
    df_historical_full_pre_fe = df_historical_full_pre_fe.drop(columns=[FUTURE_TARGET])
    print(f"Initial Train Final data shape (historical data excluding last {TARGET_SHIFT} periods): {df_train_final_pre_fe.shape}")
    print("\nCombining full historical and prediction raw data for feature engineering...")
    df_combined_for_fe = pd.concat([df_historical_full_pre_fe, df_predict_raw], ignore_index=True)
    df_combined_for_fe = df_combined_for_fe.sort_values(by=['customer_id', 'product_id', 'fecha'])
    print(f"Combined data shape for Feature Engineering: {df_combined_for_fe.shape}")
    print("\nApplying feature engineering on combined data...")
    df_combined_fe = run_feature_engineering_steps_v2(df_combined_for_fe, feature_engineering_steps_v2)
    print("\nSeparating featured data back into Train and Predict sets...")
    df_combined_fe['fecha'] = pd.to_datetime(df_combined_fe['fecha'])
    df_train_final_pre_fe['fecha'] = pd.to_datetime(df_train_final_pre_fe['fecha'])
    cols_to_merge = [col for col in df_combined_fe.columns if col not in [TARGET, FUTURE_TARGET]]
    df_train_final = pd.merge(
        df_train_final_pre_fe[['customer_id', 'product_id', 'fecha', TARGET, FUTURE_TARGET]],
        df_combined_fe[cols_to_merge],
        on=['customer_id', 'product_id', 'fecha'],
        how='left'
    )
    df_predict_final = df_combined_fe[df_combined_fe['fecha'] == pd.to_datetime(str(PREDICTION_PERIOD), format='%Y%m')].copy()
    if FUTURE_TARGET in df_predict_final.columns:
         df_predict_final = df_predict_final.drop(columns=[FUTURE_TARGET])
    print(f"Train final data shape (after feature engineering): {df_train_final.shape}")
    print(f"Predict final data shape (for {PREDICTION_PERIOD}): {df_predict_final.shape}")
    if df_train_final.empty:
        print("Critical Warning: Final training data is empty. Cannot train a model.")
    if df_predict_final.empty:
        print("Warning: Final prediction data for target period is empty.")
    cols_to_exclude_from_cat = ['customer_id', 'product_id', 'periodo', 'fecha', TARGET, FUTURE_TARGET] + [col for col in df_train_final.columns if df_train_final[col].dtype in ['int64', 'float64']]
    potential_categorical_cols = [col for col in df_train_final.columns if df_train_final[col].dtype in ['object', 'category'] and col not in cols_to_exclude_from_cat]
    print(f"\nConverting columns to 'category' dtype for LGBM: {potential_categorical_cols}")
    for col in potential_categorical_cols:
        if col in df_train_final.columns:
            df_train_final[col] = df_train_final[col].astype('category')
        else:
            print(f"Warning: Categorical column '{col}' not found in train final data. Skipping.")
        if col in df_predict_final.columns:
             if col in df_train_final.columns:
                  df_predict_final[col] = df_predict_final[col].astype('category').cat.set_categories(df_train_final[col].cat.categories)
        else:
             print(f"Warning: Categorical column '{col}' not found in predict final data. Skipping.")
    if not pd.api.types.is_datetime64_any_dtype(df_train_final['fecha']):
         df_train_final['fecha'] = pd.to_datetime(df_train_final['fecha'])
    if not pd.api.types.is_datetime64_any_dtype(df_predict_final['fecha']):
         df_predict_final['fecha'] = pd.to_datetime(df_predict_final['fecha'])
    print("\nAdding date features...")
    df_train_final['month'] = df_train_final['fecha'].dt.month
    df_train_final['year'] = df_train_final['fecha'].dt.year
    df_train_final['day'] = df_train_final['fecha'].dt.day
    df_train_final['dayofweek'] = df_train_final['fecha'].dt.dayofweek
    df_predict_final['month'] = df_predict_final['fecha'].dt.month
    df_predict_final['year'] = df_predict_final['fecha'].dt.year
    df_predict_final['day'] = df_predict_final['fecha'].dt.day
    df_predict_final['dayofweek'] = df_predict_final['fecha'].dt.dayofweek
    print("\nEnsuring ID columns are category type...")
    id_cols_to_cat = ['customer_id', 'product_id', 'periodo']
    for col in id_cols_to_cat:
        if col in df_train_final.columns:
            df_train_final[col] = df_train_final[col].astype('category')
        else:
            print(f"Warning: ID column '{col}' not found in train final data for category conversion. Skipping.")
        if col in df_predict_final.columns:
             if col in df_train_final.columns:
                  df_predict_final[col] = df_predict_final[col].astype('category').cat.set_categories(df_train_final[col].cat.categories)
             else:
                  df_predict_final[col] = df_predict_final[col].astype('category')
        else:
             print(f"Warning: ID column '{col}' not found in predict final data for category conversion. Skipping.")
    def handle_missing_values_final(df):
        num_cols_to_fill = [col for col in df.select_dtypes(include=np.number).columns if col != FUTURE_TARGET]
        cat_cols = df.select_dtypes(include='category').columns
        cols_to_leave_as_nan = [col for col in df.columns if 'lag_' in col or 'moving_avg' in col]
        num_cols_for_zero_fill = [col for col in num_cols_to_fill if col not in cols_to_leave_as_nan]
        print(f"Filling numeric NaNs with 0 (excluding lags/moving_avg): {num_cols_for_zero_fill}")
        df[num_cols_for_zero_fill] = df[num_cols_for_zero_fill].fillna(0)
        print(f"Filling categorical NaNs with 'missing': {cat_cols.tolist()}")
        for col in cat_cols:
             if 'missing' not in df[col].cat.categories:
                 df[col] = df[col].cat.add_categories('missing')
             df[col] = df[col].fillna('missing')
        return df
    df_train_final = handle_missing_values_final(df_train_final)
    df_predict_final = handle_missing_values_final(df_predict_final)
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