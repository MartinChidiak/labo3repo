import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import nbformat
import pandas as pd
import numpy as np
import os
import warnings
from scipy.stats import linregress

GCS_BUCKET_PATH = '/home/chidiakmartin/gcs-bucket'

SELL_IN_PATH = os.path.join(GCS_BUCKET_PATH, 'sell-in.txt')
PRODUCTOS_PATH = os.path.join(GCS_BUCKET_PATH, 'tb_productos.txt')
STOCKS_PATH = os.path.join(GCS_BUCKET_PATH, 'tb_stocks.txt')  
EVENTOS_PATH = os.path.join(GCS_BUCKET_PATH, 'eventos_macro_arg_2017_2019.txt')  
CHECKPOINTS_DIR = os.path.join(GCS_BUCKET_PATH, 'checkpoints')

def cargar_y_combinar_datos(sell_in_path, productos_path, stocks_path, **params):
    sell_in = pd.read_csv(sell_in_path, delimiter='\t')
    productos = pd.read_csv(productos_path, delimiter='\t')
    stocks = pd.read_csv(stocks_path, delimiter='\t')
    productos = productos.drop_duplicates(subset=['product_id'])
    df = sell_in.merge(stocks, on=['periodo', 'product_id'], how='left')
    df = df.merge(productos, on='product_id', how='left')
    return df

def optimize_dtypes(df):
    for col, dtype in [
        ('periodo', 'int32'),
        ('customer_id', 'int32'),
        ('product_id', 'int32'),
        ('plan_precios_cuidados', 'uint8'),
        ('cust_request_qty', 'float32'),
        ('cust_request_tn', 'float32'),
        ('tn', 'float32'),
        ('stock_final', 'float32'),
        ('sku_size', 'float32')
    ]:
        if col in df.columns:
            df[col] = df[col].astype(dtype)
    for col in ['cat1', 'cat2', 'cat3', 'brand']:
        if col in df.columns:
            df[col] = df[col].astype('category')
    return df


def transformar_periodo(df, **params):
    df['fecha'] = pd.to_datetime(df['periodo'].astype(str), format='%Y%m')
    return df

def generar_combinaciones_por_periodo(df, **params):
    df_copy = df.copy()
    if not isinstance(df_copy['fecha'].dtype, pd.PeriodDtype):
        df_copy['fecha'] = df_copy['fecha'].dt.to_period('M')
    customer_activity = df_copy.groupby('customer_id')['fecha'].agg(['min', 'max']).reset_index()
    customer_activity.columns = ['customer_id', 'cust_primer_fecha', 'cust_ultima_fecha']
    product_activity = df_copy.groupby('product_id')['fecha'].agg(['min', 'max']).reset_index()
    product_activity.columns = ['product_id', 'prod_primer_fecha', 'prod_ultima_fecha']
    combinaciones_fechas = customer_activity.merge(product_activity, how='cross')
    combinaciones_solapadas = combinaciones_fechas[
        (combinaciones_fechas['cust_primer_fecha'] <= combinaciones_fechas['prod_ultima_fecha']) &
        (combinaciones_fechas['cust_ultima_fecha'] >= combinaciones_fechas['prod_primer_fecha'])
    ].copy()
    combinaciones_solapadas['solapamiento_inicio'] = combinaciones_solapadas[['cust_primer_fecha', 'prod_primer_fecha']].max(axis=1)
    combinaciones_solapadas['solapamiento_fin'] = combinaciones_solapadas[['cust_ultima_fecha', 'prod_ultima_fecha']].min(axis=1)
    threshold_period = pd.Period('2019-05', freq='M')
    extension_end_period = pd.Period('2019-12', freq='M')
    condition_extend = (combinaciones_solapadas['cust_ultima_fecha'] >= threshold_period) & \
                       (combinaciones_solapadas['prod_ultima_fecha'] >= threshold_period)
    combinaciones_solapadas['solapamiento_fin'] = combinaciones_solapadas.apply(
        lambda row: extension_end_period if condition_extend[row.name] and row['solapamiento_fin'] < extension_end_period else row['solapamiento_fin'],
        axis=1
    )
    filas_expandidas = []
    for index, row in combinaciones_solapadas.iterrows():
        customer_id = row['customer_id']
        product_id = row['product_id']
        start_period = row['solapamiento_inicio']
        end_period = row['solapamiento_fin']
        period_range = pd.period_range(start=start_period, end=end_period, freq='M')
        for period in period_range:
            filas_expandidas.append({
                'customer_id': customer_id,
                'product_id': product_id,
                'fecha': period
            })
    combinaciones_por_periodo = pd.DataFrame(filas_expandidas)
    combinaciones_por_periodo['fecha'] = combinaciones_por_periodo['fecha'].astype('period[M]')
    return combinaciones_por_periodo

def merge_with_original_data(combinations_df, original_df, **params):
    if isinstance(combinations_df['fecha'].dtype, pd.PeriodDtype):
         combinations_df = combinations_df.copy()
         combinations_df['fecha'] = combinations_df['fecha'].dt.to_timestamp()
    df_completo_merged = pd.merge(
        combinations_df,
        original_df,
        on=['customer_id', 'product_id', 'fecha'],
        how='left'
    )
    # Rellenar NaNs numéricos con 0
    num_cols = df_completo_merged.select_dtypes(include=[np.number]).columns
    df_completo_merged[num_cols] = df_completo_merged[num_cols].fillna(0)
    return df_completo_merged

def fill_missing_product_info(df_merged, products_df=None, **params):
    if products_df is None:
        products_df = params.get("products_df")
    df_filled = df_merged.copy()
    product_cols_to_fill = ['cat1', 'cat2', 'cat3', 'brand', 'sku_size']
    df_filled = pd.merge(
        df_filled,
        products_df[['product_id'] + product_cols_to_fill],
        on='product_id',
        how='left',
        suffixes=('', '_from_productos')
    )
    for col in product_cols_to_fill:
        col_original = col
        col_from_productos = f'{col}_from_productos'
        if col_original in df_filled.columns and col_from_productos in df_filled.columns:
            df_filled[col_original].fillna(df_filled[col_from_productos], inplace=True)
            df_filled.drop(columns=[col_from_productos], inplace=True)
    return df_filled

# --- FUNCIONES DE FEATURE ENGINEERING (igual que antes, con **params) ---

def calculate_brand_loyalty(df, **params):
    brand_cat_counts = df.groupby(['customer_id', 'cat3', 'brand']).size().reset_index(name='brand_cat_count')
    cat_counts = df.groupby(['customer_id', 'cat3']).size().reset_index(name='cat_count')
    merged_counts = brand_cat_counts.merge(cat_counts, on=['customer_id', 'cat3'], how='left')
    merged_counts['brand_loyalty'] = merged_counts['brand_cat_count'] / merged_counts['cat_count']
    df_with_loyalty = df.merge(
        merged_counts[['customer_id', 'cat3', 'brand', 'brand_loyalty']],
        on=['customer_id', 'cat3', 'brand'],
        how='left'
    )
    df_with_loyalty['brand_loyalty'] = df_with_loyalty['brand_loyalty'].fillna(0)
    return df_with_loyalty

def add_customer_category_avg_tn(df, **params):
    df['cliente_categoria'] = df['customer_id'].astype(str) + "_" + df['cat3'].astype(str)
    media_tn_cliente_cat3 = df.groupby(['customer_id', 'cat3'])['tn'].mean().reset_index(name='tn_promedio_cliente_cat3')
    df_with_avg_tn = df.merge(
        media_tn_cliente_cat3,
        on=['customer_id', 'cat3'],
        how='left'
    )
    return df_with_avg_tn

def calculate_product_moving_avg(df, **params):
    df_with_moving_avg = df.copy()
    df_with_moving_avg = df_with_moving_avg.sort_values(by=['product_id', 'fecha'])
    df_with_moving_avg['tn_moving_avg_3m'] = df_with_moving_avg.groupby('product_id')['tn'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    return df_with_moving_avg

def add_macro_event_flag(df, event_file_path=None, **params):
    if event_file_path is None:
        event_file_path = params.get("event_file_path")
    if 'fecha' not in df.columns:
        raise ValueError("Input DataFrame must contain a 'fecha' column.")
    try:
        events_df = pd.read_csv(event_file_path, delimiter='|', skipinitialspace=True)
        events_df.columns = events_df.columns.str.strip()
        if 'Fecha' not in events_df.columns:
            raise ValueError(f"'{event_file_path}' must contain a 'Fecha' column.")
        try:
            events_df['Fecha'] = pd.to_datetime(events_df['Fecha'].str.strip())
        except Exception as e:
            raise ValueError(f"Could not parse 'Fecha' column in '{event_file_path}'. Ensure it's in YYYY-MM-DD format. Error: {e}")
        event_months = events_df['Fecha'].dt.strftime('%Y-%m').unique()
        event_months_set = set(event_months)
    except FileNotFoundError:
        raise FileNotFoundError(f"Macro event file not found at: {event_file_path}")
    except Exception as e:
        raise RuntimeError(f"Error processing macro event file '{event_file_path}': {e}")
    df_copy = pd.DataFrame({'fecha': df['fecha']})
    if isinstance(df_copy['fecha'].dtype, pd.PeriodDtype):
        df_copy['fecha_ym_str'] = df_copy['fecha'].astype(str)
    else:
        df_copy['fecha_ym_str'] = df_copy['fecha'].dt.strftime('%Y-%m')
    df_copy['is_macro_event'] = df_copy['fecha_ym_str'].isin(event_months_set)
    df['is_macro_event'] = df_copy['is_macro_event']
    return df

def calculate_tn_percentage_change(df, **params):
    df_copy = df.copy()
    df_copy = df_copy.sort_values(by=['customer_id', 'product_id', 'fecha'])
    df_copy['tn_prev'] = df_copy.groupby(['customer_id', 'product_id'])['tn'].shift(1)
    df_copy['tn_pct_change'] = np.nan
    mask_prev_zero_curr_pos = (df_copy['tn_prev'] == 0) & (df_copy['tn'] > 0)
    df_copy.loc[mask_prev_zero_curr_pos, 'tn_pct_change'] = df_copy.loc[mask_prev_zero_curr_pos, 'tn']
    mask_both_zero = (df_copy['tn_prev'] == 0) & (df_copy['tn'] == 0)
    df_copy.loc[mask_both_zero, 'tn_pct_change'] = 0
    mask_curr_zero_prev_pos = (df_copy['tn'] == 0) & (df_copy['tn_prev'] > 0)
    df_copy.loc[mask_curr_zero_prev_pos, 'tn_pct_change'] = -1
    mask_normal_case = (df_copy['tn_prev'] > 0) & (df_copy['tn'] > 0)
    df_copy.loc[mask_normal_case, 'tn_pct_change'] = (df_copy.loc[mask_normal_case, 'tn'] - 
                                                     df_copy.loc[mask_normal_case, 'tn_prev']) / \
                                                     df_copy.loc[mask_normal_case, 'tn_prev']
    df_copy = df_copy.drop(columns=['tn_prev'])
    return df_copy

def calculate_months_since_last_purchase(df, **params):
    df_copy = df.copy()
    df_copy = df_copy.sort_values(by=['customer_id', 'product_id', 'fecha'])
    df_copy['had_purchase'] = df_copy['tn'] > 0
    last_purchase_dates = df_copy[df_copy['had_purchase']].groupby(['customer_id', 'product_id'])['fecha'].max()
    last_purchase_dates = last_purchase_dates.reset_index()
    last_purchase_dates.columns = ['customer_id', 'product_id', 'last_purchase_date']
    df_copy = df_copy.merge(last_purchase_dates, on=['customer_id', 'product_id'], how='left')
    if isinstance(df_copy['fecha'].dtype, pd.PeriodDtype):
        df_copy['months_since_last_purchase'] = (df_copy['fecha'] - df_copy['last_purchase_date']).apply(
            lambda x: x.n if pd.notnull(x) else 999
        )
    else:
        df_copy['months_since_last_purchase'] = (
            (df_copy['fecha'].dt.year - df_copy['last_purchase_date'].dt.year) * 12 +
            (df_copy['fecha'].dt.month - df_copy['last_purchase_date'].dt.month)
        ).fillna(999)
    df_copy = df_copy.drop(columns=['had_purchase', 'last_purchase_date'])
    return df_copy

def calculate_customer_category_count(df, **params):
    df_copy = df.copy()
    category_counts = df_copy.groupby('customer_id')['cat1'].nunique().reset_index()
    category_counts.columns = ['customer_id', 'num_categories_per_customer']
    df_copy = df_copy.merge(category_counts, on='customer_id', how='left')
    return df_copy

def calculate_weighted_tn_sum(df, window_size=3, **params):
    window_size = params.get("window_size", window_size)
    df_copy = df.copy()
    df_copy = df_copy.sort_values(by=['customer_id', 'product_id', 'fecha'])
    if window_size == 3:
        weights = np.array([0.2, 0.3, 0.5])
    else:
        weights = np.linspace(1, 0.1, window_size)
        weights = weights / weights.sum()
        weights = weights[::-1]
    def rolling_weighted_sum(values):
        if len(values) < len(weights):
            adjusted_weights = weights[-len(values):]
            adjusted_weights = adjusted_weights / adjusted_weights.sum()
            return np.sum(values * adjusted_weights)
        else:
            return np.sum(values * weights)
    df_copy['weighted_tn_sum'] = df_copy.groupby(['customer_id', 'product_id'])['tn'].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).apply(rolling_weighted_sum, raw=True)
    )
    return df_copy

def calculate_demand_growth_rate_diff(df, **params):
    df_copy = df.copy()
    df_copy = df_copy.sort_values(by=['customer_id', 'product_id', 'fecha'])
    df_copy['tn_prev'] = df_copy.groupby(['customer_id', 'product_id'])['tn'].shift(1)
    df_copy['growth_rate'] = np.where(
        df_copy['tn_prev'] != 0,
        (df_copy['tn'] - df_copy['tn_prev']) / df_copy['tn_prev'],
        np.nan
    )
    df_copy['demand_growth_rate_diff'] = df_copy.groupby(['customer_id', 'product_id'])['growth_rate'].diff()
    df_copy = df_copy.drop(columns=['tn_prev', 'growth_rate'])
    return df_copy

def generar_lags_por_combinacion(df, columnas_para_lag=None, num_lags=12, **params):
    if columnas_para_lag is None:
        columnas_para_lag = params.get("columnas_para_lag")
    num_lags = params.get("num_lags", num_lags)
    required_cols = ['customer_id', 'product_id', 'fecha']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"El DataFrame debe contener las columnas: {missing}")
    df_con_lags = df.copy()
    df_con_lags = df_con_lags.sort_values(by=['customer_id', 'product_id', 'fecha'])
    grouped = df_con_lags.groupby(['customer_id', 'product_id'])
    for col in columnas_para_lag:
        if col not in df_con_lags.columns:
            print(f"Advertencia: La columna '{col}' no se encontró en el DataFrame. Saltando.")
            continue
        for i in range(1, num_lags + 1):
            lag_col_name = f'{col}_lag_{i}'
            df_con_lags[lag_col_name] = grouped[col].shift(i)
    return df_con_lags

def add_total_tn_per_product(df, **params):
    # Ordena por producto y fecha para asegurar el orden temporal
    df = df.sort_values(['product_id', 'fecha'])
    # Suma acumulada de tn por producto, SIN incluir la fila actual (shift(1))
    df['total_tn_per_product_to_date'] = (
        df.groupby('product_id')['tn']
        .cumsum()
        .shift(1)
    )
    # Para la primera aparición de cada producto, el valor será NaN; lo rellenamos con 0
    df['total_tn_per_product_to_date'] = df['total_tn_per_product_to_date'].fillna(0)
    return df

def add_rolling_statistics_features(df, columns=None, windows=None, stats=None, **params):
    """
    Calcula varias estadísticas de ventana móvil para columnas especificadas.
    Por defecto, calcula la media y la desviación estándar para 'tn' con ventanas de 3 y 6.
    """
    df_copy = df.copy()
    df_copy = df_copy.sort_values(by=['product_id', 'customer_id', 'fecha'])

    columns = params.get("rolling_columns", columns if columns is not None else ['tn'])
    windows = params.get("rolling_windows", windows if windows is not None else [3, 6])
    stats = params.get("rolling_stats", stats if stats is not None else ['mean', 'std'])

    grouped = df_copy.groupby(['product_id', 'customer_id'])

    for col in columns:
        if col not in df_copy.columns:
            warnings.warn(f"Advertencia: La columna '{col}' no se encontró en el DataFrame para estadísticas de ventana móvil. Saltando.")
            continue
        for window in windows:
            rolling_window = grouped[col].rolling(window=window, min_periods=1)
            for stat in stats:
                feature_name = f'{col}_rolling_{stat}_{window}m'
                if stat == 'mean':
                    df_copy[feature_name] = rolling_window.mean().reset_index(level=[0,1], drop=True)
                elif stat == 'std':
                    df_copy[feature_name] = rolling_window.std().reset_index(level=[0,1], drop=True)
                elif stat == 'min':
                    df_copy[feature_name] = rolling_window.min().reset_index(level=[0,1], drop=True)
                elif stat == 'max':
                    df_copy[feature_name] = rolling_window.max().reset_index(level=[0,1], drop=True)
                elif stat == 'median':
                    df_copy[feature_name] = rolling_window.median().reset_index(level=[0,1], drop=True)
                elif stat == 'skew':
                    df_copy[feature_name] = rolling_window.skew().reset_index(level=[0,1], drop=True)
                elif stat == 'zscore':
                    rolling_mean = rolling_window.mean().reset_index(level=[0,1], drop=True)
                    rolling_std = rolling_window.std().reset_index(level=[0,1], drop=True)
                    df_copy[feature_name] = (df_copy[col] - rolling_mean) / (rolling_std + 1e-6)
                else:
                    warnings.warn(f"Estadística '{stat}' no soportada para ventana móvil. Saltando.")
    return df_copy

def add_exponential_moving_average_features(df, columns=None, spans=None, **params):
    """
    Calcula la media móvil exponencial (EMA) para columnas específicas.
    Por defecto, calcula EMA para 'tn' con spans de 3 y 6.
    """
    df_copy = df.copy()
    df_copy = df_copy.sort_values(by=['product_id', 'customer_id', 'fecha'])

    columns = params.get("ema_columns", columns if columns is not None else ['tn'])
    spans = params.get("ema_spans", spans if spans is not None else [3, 6])

    grouped = df_copy.groupby(['product_id', 'customer_id'])

    for col in columns:
        if col not in df_copy.columns:
            warnings.warn(f"Advertencia: La columna '{col}' no se encontró en el DataFrame para EMA. Saltando.")
            continue
        for span in spans:
            df_copy[f'{col}_ema_{span}m'] = grouped[col].transform(
                lambda x: x.ewm(span=span, adjust=False).mean()
            )
    return df_copy

def add_trend_features(df, columns=None, windows=None, **params):
    """
    Calcula la tendencia (pendiente de regresión lineal) sobre una ventana móvil.
    Por defecto, calcula la tendencia para 'tn' con ventanas de 3 y 6.
    Requiere `scipy.stats.linregress`.
    """
    df_copy = df.copy()
    df_copy = df_copy.sort_values(by=['product_id', 'customer_id', 'fecha'])

    columns = params.get("trend_columns", columns if columns is not None else ['tn'])
    windows = params.get("trend_windows", windows if windows is not None else [3, 6])

    def calculate_rolling_trend(series):
        if len(series) < 2: # Need at least 2 points for a line
            return np.nan
        # Filter out NaN values, as linregress can't handle them
        valid_series = series.dropna()
        if len(valid_series) < 2:
            return np.nan
        return linregress(np.arange(len(valid_series)), valid_series)[0]

    grouped = df_copy.groupby(['product_id', 'customer_id'])

    for col in columns:
        if col not in df_copy.columns:
            warnings.warn(f"Advertencia: La columna '{col}' no se encontró en el DataFrame para tendencia. Saltando.")
            continue
        for window in windows:
            df_copy[f'{col}_trend_{window}m'] = grouped[col].transform(
                lambda x: x.rolling(window=window, min_periods=2).apply(calculate_rolling_trend, raw=False)
            )
    return df_copy

def add_difference_features(df, columns=None, periods=None, **params):
    """
    Calcula la diferencia entre el valor actual y el valor de 'n' períodos atrás.
    Por defecto, calcula la diferencia para 'tn' con períodos de 1 y 3.
    """
    df_copy = df.copy()
    df_copy = df_copy.sort_values(by=['product_id', 'customer_id', 'fecha'])

    columns = params.get("diff_columns", columns if columns is not None else ['tn'])
    periods = params.get("diff_periods", periods if periods is not None else [1, 3])

    grouped = df_copy.groupby(['product_id', 'customer_id'])

    for col in columns:
        if col not in df_copy.columns:
            warnings.warn(f"Advertencia: La columna '{col}' no se encontró en el DataFrame para diferencias. Saltando.")
            continue
        for period in periods:
            df_copy[f'{col}_diff_{period}m'] = grouped[col].diff(period)
    return df_copy

def add_total_category_sales(df, categories=None, measure_col='tn', div_by_row=False, **params):
    """
    Calcula la suma total de una columna de medida (ej. 'tn') por fecha y categoría.
    Opcionalmente, divide el valor de la fila por la suma total de la categoría en esa fecha.
    Por defecto, usa 'tn' y 'cat1'.
    """
    df_copy = df.copy()
    df_copy = df_copy.sort_values(['fecha'])

    categories = params.get("category_cols", categories if categories is not None else ['cat1', 'cat2', 'cat3'])
    measure_col = params.get("measure_column", measure_col)
    div_by_row = params.get("divide_by_row", div_by_row)

    if measure_col not in df_copy.columns:
        raise ValueError(f"La columna de medida '{measure_col}' no se encontró en el DataFrame.")

    for cat_col in categories:
        if cat_col not in df_copy.columns:
            warnings.warn(f"Advertencia: La columna de categoría '{cat_col}' no se encontró. Saltando.")
            continue
        feature_name = f"{measure_col}_{cat_col}_vendidas"
        df_copy[feature_name] = (
            df_copy.groupby(['fecha', cat_col])[measure_col]
            .transform('sum')
        )
        if div_by_row:
            df_copy[feature_name] = 1000 * df_copy[measure_col] / (df_copy[feature_name] + 1e-6) # Add epsilon to avoid division by zero
    return df_copy

def add_customer_product_total_weights(df, **params):
    """
    Calcula el peso relativo de las ventas de un cliente y un producto
    en comparación con las ventas totales de esa fecha.
    """
    df_copy = df.copy()
    df_copy = df_copy.sort_values(['fecha'])

    # Peso por cliente
    df_copy['tn_customer_vendidas'] = (
        df_copy.groupby(['fecha', 'customer_id'])['tn']
        .transform('sum')
    )
    df_copy['tn_total_vendidas_fecha'] = (
        df_copy.groupby('fecha')['tn']
        .transform('sum')
    )
    df_copy['customer_weight'] = df_copy['tn_customer_vendidas'] / (df_copy['tn_total_vendidas_fecha'] + 1e-6)

    # Peso por producto
    df_copy['tn_product_vendidas'] = (
        df_copy.groupby(['fecha', 'product_id'])['tn']
        .transform('sum')
    )
    df_copy['product_weight'] = df_copy['tn_product_vendidas'] / (df_copy['tn_total_vendidas_fecha'] + 1e-6)

    # Eliminar columnas intermedias si no se necesitan en el resultado final
    df_copy = df_copy.drop(columns=['tn_customer_vendidas', 'tn_total_vendidas_fecha', 'tn_product_vendidas'])
    return df_copy

def add_interaction_features(df, column_pairs=None, interaction_type='product', **params):
    """
    Crea características de interacción entre pares de columnas (multiplicación o división).
    Por defecto, crea interacciones de producto para ('tn', 'sku_size').
    """
    df_copy = df.copy()
    
    column_pairs = params.get("interaction_column_pairs", column_pairs if column_pairs is not None else [('tn', 'sku_size')])
    interaction_type = params.get("interaction_type", interaction_type)

    if interaction_type not in ['product', 'division']:
        raise ValueError("El 'interaction_type' debe ser 'product' o 'division'.")

    for col1, col2 in column_pairs:
        if col1 not in df_copy.columns or col2 not in df_copy.columns:
            warnings.warn(f"Advertencia: Una o ambas columnas '{col1}', '{col2}' no se encontraron para interacción. Saltando.")
            continue
        if interaction_type == 'product':
            df_copy[f"{col1}_prod_{col2}"] = df_copy[col1] * df_copy[col2]
        elif interaction_type == 'division':
            df_copy[f"{col1}_div_{col2}"] = df_copy[col1] / (df_copy[col2] + 1e-6) # Evitar división por cero
    return df_copy