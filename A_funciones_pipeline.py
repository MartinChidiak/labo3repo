import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import nbformat
import numpy as np
import os

GCS_BUCKET_PATH = '/home/chidiakmartin/gcs-bucket'

SELL_IN_PATH = os.path.join(GCS_BUCKET_PATH, 'sell-in.txt')
PRODUCTOS_PATH = os.path.join(GCS_BUCKET_PATH, 'tb_productos.txt')
STOCKS_PATH = os.path.join(GCS_BUCKET_PATH, 'tb_stocks.txt')  
EVENTOS_PATH = os.path.join(GCS_BUCKET_PATH, 'eventos_macro_arg_2017_2019.txt')  
CHECKPOINTS_DIR = os.path.join(GCS_BUCKET_PATH, 'checkpoints')


# def cargar_y_combinar_datos():
#     # Cargar archivos
#     sell_in = pd.read_csv(SELL_IN_PATH, delimiter='\t')
#     productos = pd.read_csv(PRODUCTOS_PATH, delimiter='\t')
#     stocks = pd.read_csv(STOCKS_PATH, delimiter='\t')
#     #drop duplicates in productos
#     productos = productos.drop_duplicates(subset=['product_id'])
#     ## Unir Datasets
#     df = sell_in.merge(stocks, on=['periodo', 'product_id'], how='left')
#     df = df.merge(productos, on='product_id', how='left')
#     return df

def cargar_y_combinar_datos(sell_in_path, productos_path, stocks_path):
    sell_in = pd.read_csv(sell_in_path, delimiter='\t')
    productos = pd.read_csv(productos_path, delimiter='\t')
    stocks = pd.read_csv(stocks_path, delimiter='\t')
    productos = productos.drop_duplicates(subset=['product_id'])
    df = sell_in.merge(stocks, on=['periodo', 'product_id'], how='left')
    df = df.merge(productos, on='product_id', how='left')
    return df

def transformar_periodo(df):
    """
    Convierte la columna 'periodo' a tipo datetime.
    """
    df['fecha'] = pd.to_datetime(df['periodo'].astype(str), format='%Y%m')
    return df

def generar_lags_por_combinacion(df, columnas_para_lag, num_lags=12):
    """
    Genera variables lag para columnas específicas dentro de cada grupo
    (customer_id, product_id). Requiere las columnas 'customer_id',
    'product_id' y 'fecha'.

    Args:
        df (pd.DataFrame): El DataFrame de entrada que contiene las series
                           temporales (requiere 'customer_id', 'product_id', 'fecha').
        columnas_para_lag (list): Lista de nombres de columnas numéricas para
                                  las que generar lags.
        num_lags (int): El número de lags a generar (por defecto es 12).

    Returns:
        pd.DataFrame: El DataFrame original con las nuevas columnas lag añadidas.
                      Las primeras filas de cada grupo para las columnas lag
                      contendrán valores NaN.
    Raises:
        ValueError: Si faltan las columnas 'customer_id', 'product_id' o 'fecha'.
    """
    # Validar la presencia de columnas requeridas
    required_cols = ['customer_id', 'product_id', 'fecha']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"El DataFrame debe contener las columnas: {missing}")

    df_con_lags = df.copy() # Trabajar con una copia para no modificar el original

    # Asegurarse de que el DataFrame esté ordenado por la clave de la serie temporal y fecha
    # Esto es CRUCIAL para que shift() funcione correctamente dentro de los grupos
    df_con_lags = df_con_lags.sort_values(by=['customer_id', 'product_id', 'fecha'])

    # Agrupar por la clave de la serie temporal
    grouped = df_con_lags.groupby(['customer_id', 'product_id'])

    # Iterar sobre cada columna y cada número de lag
    for col in columnas_para_lag:
        # Verificar si la columna para laggear existe en el DataFrame
        if col not in df_con_lags.columns:
            print(f"Advertencia: La columna '{col}' no se encontró en el DataFrame. Saltando.")
            continue # Saltar a la siguiente columna si no existe

        # Generar los lags para la columna actual
        for i in range(1, num_lags + 1):
            # Nombre de la nueva columna lag
            lag_col_name = f'{col}_lag_{i}'

            # Aplicar shift dentro de cada grupo para generar el lag
            df_con_lags[lag_col_name] = grouped[col].shift(i)

    return df_con_lags 

def generar_combinaciones_por_periodo(df):
    """
    Genera todas las posibles combinaciones de customer_id, product_id, y fecha
    para los periodos donde ambos (cliente y producto) estuvieron activos.
    Extiende el rango hasta 2019-12 para combinaciones activas en 2019-05 o después.

    Args:
        df (pd.DataFrame): DataFrame de entrada con al menos 'customer_id',
                           'product_id', y 'fecha' (as datetime or Period[M]).

    Returns:
        pd.DataFrame: DataFrame con todas las combinaciones únicas de
                      (customer_id, product_id, fecha) dentro de sus periodos activos,
                      extended for specific cases, with 'fecha' as Period[M].
    """
    df_copy = df.copy()

    # Ensure 'fecha' is Period[M] for consistency and correct period arithmetic
    if not isinstance(df_copy['fecha'].dtype, pd.PeriodDtype):
        print("Converting 'fecha' column to Period[M] in generar_combinaciones_por_periodo")
        df_copy['fecha'] = df_copy['fecha'].dt.to_period('M')


    # Calcular el primer y último periodo activo para cada cliente
    customer_activity = df_copy.groupby('customer_id')['fecha'].agg(['min', 'max']).reset_index()
    customer_activity.columns = ['customer_id', 'cust_primer_fecha', 'cust_ultima_fecha']

    # Calcular el primer y último periodo activo para cada producto
    product_activity = df_copy.groupby('product_id')['fecha'].agg(['min', 'max']).reset_index()
    product_activity.columns = ['product_id', 'prod_primer_fecha', 'prod_ultima_fecha']

    # Realizar el cross join para obtener todas las combinaciones cliente-producto con sus rangos de fechas activos
    combinaciones_fechas = customer_activity.merge(product_activity, how='cross')

    # Filtrar las combinaciones donde los rangos de fecha se solapan
    # Ensure comparison is between Period[M] and Period[M]
    combinaciones_solapadas = combinaciones_fechas[
        (combinaciones_fechas['cust_primer_fecha'] <= combinaciones_fechas['prod_ultima_fecha']) &
        (combinaciones_fechas['cust_ultima_fecha'] >= combinaciones_fechas['prod_primer_fecha'])
    ].copy()

    # Calcular las fechas de inicio y fin del solapamiento (as Period[M])
    combinaciones_solapadas['solapamiento_inicio'] = combinaciones_solapadas[['cust_primer_fecha', 'prod_primer_fecha']].max(axis=1)
    combinaciones_solapadas['solapamiento_fin'] = combinaciones_solapadas[['cust_ultima_fecha', 'prod_ultima_fecha']].min(axis=1)

    # Define the threshold period and target extension period as Period[M]
    threshold_period = pd.Period('2019-05', freq='M')
    extension_end_period = pd.Period('2019-12', freq='M')

    # Identify combinations where both customer and product were active on or after the threshold
    # Ensure comparison is between Period[M] and Period[M]
    condition_extend = (combinaciones_solapadas['cust_ultima_fecha'] >= threshold_period) & \
                       (combinaciones_solapadas['prod_ultima_fecha'] >= threshold_period)

    # For these combinations, extend the 'solapamiento_fin' up to extension_end_period
    # but only if the current 'solapamiento_fin' is before the extension_end_period
    # Ensure comparison is between Period[M] and Period[M]
    combinaciones_solapadas['solapamiento_fin'] = combinaciones_solapadas.apply(
        lambda row: extension_end_period if condition_extend[row.name] and row['solapamiento_fin'] < extension_end_period else row['solapamiento_fin'],
        axis=1
    )

    # Generate a list of periods (months) for each overlapping combination
    filas_expandidas = []
    for index, row in combinaciones_solapadas.iterrows():
        customer_id = row['customer_id']
        product_id = row['product_id']
        # Use the calculated solapamiento_inicio and potentially extended solapamiento_fin (Period[M])
        start_period = row['solapamiento_inicio']
        end_period = row['solapamiento_fin']

        # Generate period range (inclusive)
        period_range = pd.period_range(start=start_period, end=end_period, freq='M')

        # Create a new row for each period in the range
        for period in period_range:
            filas_expandidas.append({
                'customer_id': customer_id,
                'product_id': product_id,
                'fecha': period # Use 'fecha' to match your main DataFrame
            })

    # Convert the list of rows into a new DataFrame
    combinaciones_por_periodo = pd.DataFrame(filas_expandidas)

    # Ensure 'fecha' column is Period[M] dtype in the final DataFrame
    combinaciones_por_periodo['fecha'] = combinaciones_por_periodo['fecha'].astype('period[M]')

    return combinaciones_por_periodo

def merge_with_original_data(combinations_df, original_df):
    """
    Merges the DataFrame with all combinations with the original data.

    Args:
        combinations_df (pd.DataFrame): DataFrame with all possible combinations.
        original_df (pd.DataFrame): Original DataFrame with data.

    Returns:
        pd.DataFrame: Merged DataFrame with missing combinations filled (as NaN).
    """
    # Ensure date columns are in compatible format for merge
    # Assuming original_df['fecha'] is already in a suitable format (e.g., datetime or timestamp)
    # If combinations_df['fecha'] is Period[M], convert it to timestamp for merging
    if isinstance(combinations_df['fecha'].dtype, pd.PeriodDtype):
         combinations_df = combinations_df.copy()
         combinations_df['fecha'] = combinations_df['fecha'].dt.to_timestamp()

    # Assuming original_df['fecha'] is already timestamp from transformar_periodo
    # If not, you might need a conversion step here for original_df as well.

    df_completo_merged = pd.merge(
        combinations_df,
        original_df,
        on=['customer_id', 'product_id', 'fecha'],
        how='left'
    )
    return df_completo_merged

def fill_missing_product_info(df_merged, products_df):
    """
    Fills missing product-related information in the merged DataFrame
    using data from the products DataFrame.

    Args:
        df_merged (pd.DataFrame): Merged DataFrame with potential missing product info.
        products_df (pd.DataFrame): DataFrame containing product details.

    Returns:
        pd.DataFrame: DataFrame with product info filled.
    """
    df_filled = df_merged.copy()
    # Define columns to fill and their source in products_df
    # Assuming products_df has 'product_id' and columns like 'cat1', 'cat2', 'cat3', 'brand', 'sku_size'
    product_cols_to_fill = ['cat1', 'cat2', 'cat3', 'brand', 'sku_size']

    # Temporarily merge products_df to get the product info next to the merged data
    # This avoids the SettingWithCopyWarning when filling NaNs based on another column's value
    df_filled = pd.merge(
        df_filled,
        products_df[['product_id'] + product_cols_to_fill], # Select relevant columns from products
        on='product_id',
        how='left', # Use left merge to keep all rows from df_filled
        suffixes=('', '_from_productos') # Add suffix to avoid name conflicts
    )

    # Fill missing values in the original columns using the suffixed columns
    for col in product_cols_to_fill:
        col_original = col
        col_from_productos = f'{col}_from_productos'
        if col_original in df_filled.columns and col_from_productos in df_filled.columns:
            df_filled[col_original].fillna(df_filled[col_from_productos], inplace=True)
            # After filling, drop the temporary columns from products_df
            df_filled.drop(columns=[col_from_productos], inplace=True)

    return df_filled 

def encode_categorical_features(df, categorical_cols_ohe, id_cols=['customer_id', 'product_id']):
    """
    Encodes categorical features using One-Hot Encoding and converts ID columns to category dtype.

    Args:
        df (pd.DataFrame): The input DataFrame.
        categorical_cols_ohe (list): List of column names to apply One-Hot Encoding to.
        id_cols (list): List of ID column names to convert to 'category' dtype.

    Returns:
        pd.DataFrame: DataFrame with categorical features encoded.
    """
    df_encoded = df.copy()

    # Convert ID columns to category dtype
    for col in id_cols:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].astype('category')
        else:
            print(f"Warning: ID column '{col}' not found for category conversion.")

    # Apply One-Hot Encoding to specified categorical columns
    # Handle potential NaNs before OHE if necessary (e.g., fillna('') or use handle_unknown='ignore')
    # For simplicity, let's fill NaNs in OHE columns with a placeholder string
    for col in categorical_cols_ohe:
        if col in df_encoded.columns:
            if df_encoded[col].isnull().any():
                print(f"Warning: Column '{col}' contains NaNs. Filling with 'Missing' before OHE.")
                df_encoded[col] = df_encoded[col].fillna('Missing')
        else:
            print(f"Warning: OHE column '{col}' not found. Skipping OHE for this column.")
            categorical_cols_ohe.remove(col) # Remove from list to avoid error in get_dummies

    # Apply one-hot encoding
    if categorical_cols_ohe:
        df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols_ohe, dummy_na=False) # dummy_na=False excludes NaN column if NaNs are present

    return df_encoded 


def calculate_brand_loyalty(df: pd.DataFrame) -> pd.DataFrame:
  """
  Calculates brand loyalty for each customer within each category.

  Brand loyalty is defined as the proportion of a customer's purchases
  in a specific category that are of a particular brand.

  Args:
    df: pandas DataFrame with columns 'customer_id', 'cat3', and 'brand'.

  Returns:
    pandas DataFrame with an added 'brand_loyalty' column.
  """
  # Calculate count of items for each customer, category, and brand
  # Use .reset_index(name='count') to keep the result as a DataFrame
  brand_cat_counts = df.groupby(['customer_id', 'cat3', 'brand']).size().reset_index(name='brand_cat_count')

  # Calculate total count of items for each customer and category
  cat_counts = df.groupby(['customer_id', 'cat3']).size().reset_index(name='cat_count')

  # Merge counts
  merged_counts = brand_cat_counts.merge(cat_counts, on=['customer_id', 'cat3'], how='left')

  # Calculate brand loyalty
  merged_counts['brand_loyalty'] = merged_counts['brand_cat_count'] / merged_counts['cat_count']

  # Merge the calculated loyalty back to the original DataFrame
  # We merge on customer_id, cat3, and brand, and only keep the brand_loyalty column
  df_with_loyalty = df.merge(
      merged_counts[['customer_id', 'cat3', 'brand', 'brand_loyalty']],
      on=['customer_id', 'cat3', 'brand'],
      how='left'
  )

  # Fill NaN values that might occur if a customer/cat3 combination had no brand
  # This might not be necessary depending on your data, but is safer.
  df_with_loyalty['brand_loyalty'] = df_with_loyalty['brand_loyalty'].fillna(0)


  return df_with_loyalty


def add_customer_category_avg_tn(df: pd.DataFrame) -> pd.DataFrame:
  """
  Adds a 'cliente_categoria' column and calculates/merges
  the average 'tn' for each customer-category combination.

  Args:
    df: pandas DataFrame with columns 'customer_id', 'cat3', and 'tn'.

  Returns:
    pandas DataFrame with added 'cliente_categoria' and
    'tn_promedio_cliente_cat3' columns.
  """
  # Create the cliente_categoria column
  df['cliente_categoria'] = df['customer_id'].astype(str) + "_" + df['cat3'].astype(str)

  # Calculate the mean of 'tn' for each customer and category
  # Use .reset_index(name='tn_promedio_cliente_cat3') to keep the result as a DataFrame
  media_tn_cliente_cat3 = df.groupby(['customer_id', 'cat3'])['tn'].mean().reset_index(name='tn_promedio_cliente_cat3')

  # Merge the calculated mean back to the original DataFrame
  df_with_avg_tn = df.merge(
      media_tn_cliente_cat3,
      on=['customer_id', 'cat3'],
      how='left'
  )

  return df_with_avg_tn

# Define a new function calculate_product_moving_avg
def calculate_product_moving_avg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la media móvil de la columna 'tn' por producto en los últimos 3 meses.

    Requiere las columnas 'product_id', 'fecha' (como Period[M] o datetime)
    y 'tn'.

    Args:
        df (pd.DataFrame): El DataFrame de entrada que contiene los datos.
                           Debe estar ordenado por 'product_id' y 'fecha'.

    Returns:
        pd.DataFrame: El DataFrame original con la nueva columna
                      'tn_moving_avg_3m' añadida.
                      Las primeras filas de cada producto para esta columna
                      contendrán valores NaN hasta que haya 3 meses de datos.
    """
    df_with_moving_avg = df.copy()

    # Ensure the DataFrame is sorted by product_id and fecha for correct rolling calculation
    df_with_moving_avg = df_with_moving_avg.sort_values(by=['product_id', 'fecha'])

    # Calculate the 3-month rolling average for 'tn' within each product group
    # window=3 means a 3-period window (assuming monthly data based on 'fecha')
    # min_periods=1 ensures that it calculates an average even with fewer than 3 periods initially
    df_with_moving_avg['tn_moving_avg_3m'] = df_with_moving_avg.groupby('product_id')['tn'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())

    return df_with_moving_avg


def add_macro_event_flag(df, event_file_path):
    """
    Adds a boolean flag column indicating if a macro event occurred in the month
    of the 'fecha' column.

    Reads event dates from the specified text file and matches them by month.

    Args:
        df (pd.DataFrame): The input DataFrame containing a 'fecha' column
                           (datetime or Period[M] dtype).
        event_file_path (str): Path to the text file containing macro events.
                               Expected format: pipe-delimited, first column is
                               'Fecha' in YYYY-MM-DD format.

    Returns:
        pd.DataFrame: The DataFrame with the new 'is_macro_event' boolean column.
    Raises:
        FileNotFoundError: If the event_file_path does not exist.
        ValueError: If the 'Fecha' column is not found or cannot be parsed in
                    the event file.
    """
    if 'fecha' not in df.columns:
        raise ValueError("Input DataFrame must contain a 'fecha' column.")

    try:
        # Read the event file - it's pipe-delimited and may have extra spaces
        events_df = pd.read_csv(event_file_path, delimiter='|', skipinitialspace=True)

        # Clean up column names by stripping whitespace
        events_df.columns = events_df.columns.str.strip()

        if 'Fecha' not in events_df.columns:
             raise ValueError(f"'{event_file_path}' must contain a 'Fecha' column.")

        # Convert the 'Fecha' column from the event file to datetime objects
        # Handle potential errors during conversion
        try:
            events_df['Fecha'] = pd.to_datetime(events_df['Fecha'].str.strip())
        except Exception as e:
             raise ValueError(f"Could not parse 'Fecha' column in '{event_file_path}'. Ensure it's in YYYY-MM-DD format. Error: {e}")

        # Extract the year-month string (e.g., '2017-03') from event dates
        event_months = events_df['Fecha'].dt.strftime('%Y-%m').unique()

        # Create a set for efficient lookup
        event_months_set = set(event_months)

    except FileNotFoundError:
        raise FileNotFoundError(f"Macro event file not found at: {event_file_path}")
    except Exception as e:
        # Catch other potential reading/processing errors
        raise RuntimeError(f"Error processing macro event file '{event_file_path}': {e}")


    # Convert the DataFrame's 'fecha' column to a comparable year-month string format
    # Handle both datetime and Period[M] dtypes
    df_copy = pd.DataFrame({'fecha': df['fecha']}) # Work on a copy to avoid modifying original df in place

    if isinstance(df_copy['fecha'].dtype, pd.PeriodDtype):
        df_copy['fecha_ym_str'] = df_copy['fecha'].astype(str)
    else:
        df_copy['fecha_ym_str'] = df_copy['fecha'].dt.strftime('%Y-%m')

    # Create the flag column by checking if the row's month is in the set of event months
    df_copy['is_macro_event'] = df_copy['fecha_ym_str'].isin(event_months_set)
    df['is_macro_event'] = df_copy['is_macro_event']

    return df

def calculate_tn_percentage_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula el cambio porcentual de 'tn' respecto al mes anterior para cada combinación
    de customer_id y product_id.

    Args:
        df (pd.DataFrame): DataFrame que debe contener las columnas 'customer_id',
                          'product_id', 'fecha' y 'tn'.

    Returns:
        pd.DataFrame: DataFrame con una nueva columna 'tn_pct_change' que muestra
                     el cambio porcentual respecto al mes anterior.
                     - Si el valor anterior es 0 y el actual es > 0, se asigna el valor actual
                     - Si el valor anterior es 0 y el actual es 0, se asigna 0
                     - Si el valor actual es 0 y el anterior es > 0, se asigna -1
                     - Para otros casos, se calcula el cambio porcentual normal
    """
    df_copy = df.copy()
    
    # Asegurarse de que el DataFrame esté ordenado por customer_id, product_id y fecha
    df_copy = df_copy.sort_values(by=['customer_id', 'product_id', 'fecha'])
    
    # Obtener el valor anterior para cada grupo
    df_copy['tn_prev'] = df_copy.groupby(['customer_id', 'product_id'])['tn'].shift(1)
    
    # Inicializar la columna de cambio porcentual
    df_copy['tn_pct_change'] = np.nan
    
    # Caso 1: Si el valor anterior es 0 y el actual es > 0
    mask_prev_zero_curr_pos = (df_copy['tn_prev'] == 0) & (df_copy['tn'] > 0)
    df_copy.loc[mask_prev_zero_curr_pos, 'tn_pct_change'] = df_copy.loc[mask_prev_zero_curr_pos, 'tn']
    
    # Caso 2: Si el valor anterior es 0 y el actual es 0
    mask_both_zero = (df_copy['tn_prev'] == 0) & (df_copy['tn'] == 0)
    df_copy.loc[mask_both_zero, 'tn_pct_change'] = 0
    
    # Caso 3: Si el valor actual es 0 y el anterior es > 0
    mask_curr_zero_prev_pos = (df_copy['tn'] == 0) & (df_copy['tn_prev'] > 0)
    df_copy.loc[mask_curr_zero_prev_pos, 'tn_pct_change'] = -1
    
    # Caso 4: Para todos los demás casos (ninguno es 0)
    mask_normal_case = (df_copy['tn_prev'] > 0) & (df_copy['tn'] > 0)
    df_copy.loc[mask_normal_case, 'tn_pct_change'] = (df_copy.loc[mask_normal_case, 'tn'] - 
                                                     df_copy.loc[mask_normal_case, 'tn_prev']) / \
                                                     df_copy.loc[mask_normal_case, 'tn_prev']
    
    # Eliminar la columna temporal
    df_copy = df_copy.drop(columns=['tn_prev'])
    
    return df_copy


def calculate_months_since_last_purchase(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula los meses transcurridos desde la última compra del producto por cliente.
    Si no hay compras previas, se asigna un valor alto (999) para indicar que nunca se ha comprado.

    Args:
        df (pd.DataFrame): DataFrame que debe contener las columnas 'customer_id',
                          'product_id', 'fecha' y 'tn'.

    Returns:
        pd.DataFrame: DataFrame con una nueva columna 'months_since_last_purchase'
                     que indica los meses transcurridos desde la última compra.
    """
    df_copy = df.copy()
    
    # Asegurarse de que el DataFrame esté ordenado
    df_copy = df_copy.sort_values(by=['customer_id', 'product_id', 'fecha'])
    
    # Crear una columna que indique si hubo compra (tn > 0)
    df_copy['had_purchase'] = df_copy['tn'] > 0
    
    # Obtener las fechas de última compra para cada combinación cliente-producto
    last_purchase_dates = df_copy[df_copy['had_purchase']].groupby(['customer_id', 'product_id'])['fecha'].max()
    
    # Convertir a DataFrame para facilitar el merge
    last_purchase_dates = last_purchase_dates.reset_index()
    last_purchase_dates.columns = ['customer_id', 'product_id', 'last_purchase_date']
    
    # Unir con el DataFrame original
    df_copy = df_copy.merge(last_purchase_dates, on=['customer_id', 'product_id'], how='left')
    
    # Calcular la diferencia en meses
    if isinstance(df_copy['fecha'].dtype, pd.PeriodDtype):
        df_copy['months_since_last_purchase'] = (df_copy['fecha'] - df_copy['last_purchase_date']).apply(
            lambda x: x.n if pd.notnull(x) else 999
        )
    else:
        df_copy['months_since_last_purchase'] = (
            (df_copy['fecha'].dt.year - df_copy['last_purchase_date'].dt.year) * 12 +
            (df_copy['fecha'].dt.month - df_copy['last_purchase_date'].dt.month)
        ).fillna(999)
    
    # Limpiar columnas temporales
    df_copy = df_copy.drop(columns=['had_purchase', 'last_purchase_date'])
    
    return df_copy

def calculate_customer_category_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula el número de categorías distintas que compra cada cliente.
    
    Args:
        df (pd.DataFrame): DataFrame que debe contener las columnas 'customer_id' y 'cat1'.
        
    Returns:
        pd.DataFrame: DataFrame con una nueva columna 'num_categories_per_customer' que indica
                     el número de categorías distintas que compra cada cliente.
    """
    df_copy = df.copy()
    
    # Calcular el número de categorías distintas por cliente
    category_counts = df_copy.groupby('customer_id')['cat1'].nunique().reset_index()
    category_counts.columns = ['customer_id', 'num_categories_per_customer']
    
    # Unir el resultado con el DataFrame original
    df_copy = df_copy.merge(category_counts, on='customer_id', how='left')
    
    return df_copy

def calculate_weighted_tn_sum(df: pd.DataFrame, window_size: int = 3) -> pd.DataFrame:
    """
    Calcula la suma ponderada de valores 'tn' recientes, dando más peso a los valores más recientes.
    
    Args:
        df (pd.DataFrame): DataFrame que debe contener las columnas 'customer_id',
                          'product_id', 'fecha' y 'tn'.
        window_size (int): Tamaño de la ventana para calcular la suma ponderada (por defecto 3).
                          Note: Weights are currently hardcoded for window_size=3.
    
    Returns:
        pd.DataFrame: DataFrame con una nueva columna 'weighted_tn_sum' que contiene
                     la suma ponderada de los valores 'tn' recientes.
    """
    df_copy = df.copy()
    
    # Asegurarse de que el DataFrame esté ordenado por customer_id, product_id y fecha
    df_copy = df_copy.sort_values(by=['customer_id', 'product_id', 'fecha'])
    
    # Define weights for the specified window size
    if window_size == 3:
        weights = np.array([0.2, 0.3, 0.5])  # Weights for window [t-2, t-1, t]
    else:
        weights = np.linspace(1, 0.1, window_size)
        weights = weights / weights.sum()
        weights = weights[::-1]

    def rolling_weighted_sum(values):  # values is already a numpy array
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


def calculate_demand_growth_rate_diff(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la diferencia en la tasa de crecimiento de la demanda (tn) entre períodos consecutivos.
    La tasa de crecimiento se calcula como el cambio porcentual en tn, y luego se calcula la diferencia
    entre tasas de crecimiento consecutivas.

    Args:
        df (pd.DataFrame): DataFrame que debe contener las columnas 'customer_id',
                          'product_id', 'fecha' y 'tn'.

    Returns:
        pd.DataFrame: DataFrame con una nueva columna 'demand_growth_rate_diff' que muestra
                     la diferencia entre tasas de crecimiento consecutivas.
                     - Si no hay datos suficientes para calcular la tasa de crecimiento,
                       se asigna NaN
                     - Si la tasa de crecimiento anterior es NaN, la diferencia también será NaN
    """
    df_copy = df.copy()
    
    # Asegurarse de que el DataFrame esté ordenado por customer_id, product_id y fecha
    df_copy = df_copy.sort_values(by=['customer_id', 'product_id', 'fecha'])
    
    # Calcular la tasa de crecimiento para cada período
    # Primero calculamos el valor anterior de tn
    df_copy['tn_prev'] = df_copy.groupby(['customer_id', 'product_id'])['tn'].shift(1)
    
    # Calcular la tasa de crecimiento
    # Evitamos división por cero usando np.where
    df_copy['growth_rate'] = np.where(
        df_copy['tn_prev'] != 0,
        (df_copy['tn'] - df_copy['tn_prev']) / df_copy['tn_prev'],
        np.nan  # Si el valor anterior es 0, asignamos NaN
    )
    
    # Calcular la diferencia entre tasas de crecimiento consecutivas
    df_copy['demand_growth_rate_diff'] = df_copy.groupby(['customer_id', 'product_id'])['growth_rate'].diff()
    
    # Limpiar columnas temporales
    df_copy = df_copy.drop(columns=['tn_prev', 'growth_rate'])
    
    return df_copy


def calculate_cust_request_tn_anomaly(df: pd.DataFrame, std_threshold: float = 2.0) -> pd.DataFrame:
    """
    Calcula un indicador de anomalía para la columna 'cust_request_tn' basado en
    la desviación estándar de los valores históricos.

    Args:
        df (pd.DataFrame): DataFrame que debe contener las columnas 'customer_id',
                          'product_id', 'fecha' y 'cust_request_tn'.
        std_threshold (float): Número de desviaciones estándar que define el límite
                              para considerar un valor como anómalo (por defecto 2.0).

    Returns:
        pd.DataFrame: DataFrame con nuevas columnas:
                     - 'is_cust_request_tn_anomaly': booleano indicando si el valor está fuera del rango
                     - 'cust_request_tn_zscore': z-score del valor actual
                     - 'cust_request_tn_rolling_mean': media móvil de 12 meses
    """
    df_copy = df.copy()
    
    # Asegurarse de que el DataFrame esté ordenado por customer_id, product_id y fecha
    df_copy = df_copy.sort_values(by=['customer_id', 'product_id', 'fecha'])
    
    def calculate_rolling_stats(group):
        # Usar una ventana de 12 meses para calcular las estadísticas
        rolling_mean = group['cust_request_tn'].rolling(window=12, min_periods=3).mean()
        rolling_std = group['cust_request_tn'].rolling(window=12, min_periods=3).std()
        
        # Calcular z-score
        zscore = (group['cust_request_tn'] - rolling_mean) / rolling_std
        
        # Calcular los límites superior e inferior
        upper_limit = rolling_mean + (std_threshold * rolling_std)
        lower_limit = rolling_mean - (std_threshold * rolling_std)
        
        # Marcar como anómalo si está fuera de los límites
        is_anomaly = (group['cust_request_tn'] > upper_limit) | (group['cust_request_tn'] < lower_limit)
        
        return pd.DataFrame({
            'is_cust_request_tn_anomaly': is_anomaly,
            'cust_request_tn_zscore': zscore,
            'cust_request_tn_rolling_mean': rolling_mean
        })
    
    # Aplicar la función a cada grupo de cliente-producto
    stats_df = df_copy.groupby(['customer_id', 'product_id']).apply(
        lambda x: calculate_rolling_stats(x)
    ).reset_index(level=0, drop=True)
    
    # Unir las nuevas columnas al DataFrame original
    df_copy = pd.concat([df_copy, stats_df], axis=1)
    
    return df_copy