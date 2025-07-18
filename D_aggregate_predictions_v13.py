# aggregate_predictions.py

import pandas as pd
import os

# Define data paths
# Assuming this script is in the same directory as train_model.py
# and checkpoints are in a 'checkpoints' subdirectory relative to that.
# You might need to adjust CHECKPOINTS_DIR if the script location changes.

# Define data paths
#GCS_BUCKET_PATH = '/home/chidiakmartin/gcs-bucket'
GCS_BUCKET_PATH = r"C:\Users\Martin\OneDrive\Maestría\20- Laboratorio3\RepositorioLabo3"
CHECKPOINTS_DIR = os.path.join(GCS_BUCKET_PATH, 'checkpoints')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the prediction period and target name to match train_model.py
PREDICTION_PERIOD = 202002 # Must match the period used in train_model.py
TARGET = 'tn' # Must match the target name used in train_model.py

# Input files
# Assuming ListadoIDS.txt is in the same directory as the script
LIST_IDS_FILE = os.path.join(SCRIPT_DIR, 'ListadoIDS.txt')
# Use the prediction period in the predictions file name
PREDICTIONS_FILE = os.path.join(CHECKPOINTS_DIR, f'predictions_{PREDICTION_PERIOD}.csv')

# Output files - Only need to save aggregated predictions
AGGREGATED_PREDICTIONS_FILE = os.path.join(CHECKPOINTS_DIR, f'aggregated_predictions_{PREDICTION_PERIOD}.csv')
# Remove the actuals file definition: AGGREGATED_ACTUALS_FILE = os.path.join(CHECKPOINTS_DIR, 'actual_tns_201912.csv')


print("Starting prediction aggregation script (Metrics Calculation Removed)")

# 1. Read the list of product IDs
try:
    with open(LIST_IDS_FILE, 'r') as f:
        # Assuming one product ID per line
        product_ids_to_filter = [line.strip() for line in f if line.strip()]
    print(f"Read {len(product_ids_to_filter)} product IDs from {LIST_IDS_FILE}")
    if not product_ids_to_filter:
        print("Warning: ListadoIDS.txt is empty or contains no valid IDs.")
        # Exit if no IDs are provided to filter
        exit()
except FileNotFoundError:
    print(f"Error: ListadoIDS.txt not found at {LIST_IDS_FILE}")
    exit()
except Exception as e:
    print(f"Error reading ListadoIDS.txt: {e}")
    exit()


# 2. Load the predictions CSV file
try:
    df_predictions = pd.read_csv(PREDICTIONS_FILE)
    print(f"Loaded predictions data from {PREDICTIONS_FILE} with shape {df_predictions.shape}")
except FileNotFoundError:
    print(f"Error: Predictions file not found at {PREDICTIONS_FILE}. Please run train_model.py first.")
    exit()
except Exception as e:
    print(f"Error loading predictions file: {e}")
    exit()

# Ensure product_id column exists and is of a comparable type
if 'product_id' not in df_predictions.columns:
    print("Error: 'product_id' column not found in the predictions file.")
    exit()

# Convert product_id to numeric first, coercing errors, then to string, and strip '.0' if it exists
try:
    # Attempt to convert to float first to handle '.0' correctly
    df_predictions['product_id'] = pd.to_numeric(df_predictions['product_id'], errors='coerce')
    # Now convert to integer (this will remove .0 if it's a whole number float)
    df_predictions['product_id'] = df_predictions['product_id'].dropna().astype(int).astype(str)
    # Handle potential NaN values created by coercion (e.g., non-numeric IDs)
    # For simplicity here, we'll assume valid numeric IDs; adjust if non-numeric IDs are possible.
    # If you might have non-numeric IDs, a more robust approach is needed.
    # For now, assuming numeric IDs with potential .0
    df_predictions['product_id'] = df_predictions['product_id'].apply(lambda x: str(int(float(x))) if pd.notna(x) and '.' in str(float(x)) else str(x))

except ValueError:
     # If direct float conversion fails for some reason, fall back to string and strip
     print("Warning: Could not convert product_id to numeric. Falling back to string stripping.")
     df_predictions['product_id'] = df_predictions['product_id'].astype(str).str.replace(r'\.0$', '', regex=True)

# Ensure IDs from the text file match the dtype in the dataframe
# This is important for accurate filtering
product_ids_to_filter = [str(id) for id in product_ids_to_filter]

# 3. Filter predictions by product IDs
df_filtered = df_predictions[df_predictions['product_id'].isin(product_ids_to_filter)].copy()
print(f"Filtered predictions data shape: {df_filtered.shape}")

if df_filtered.empty:
    print("No data found for the specified product IDs in the predictions file.")
    # Create empty dataframe for aggregated predictions
    aggregated_predictions_df = pd.DataFrame(columns=['product_id', 'total_predicted_tn'])
    # Remove creation of aggregated_actuals_df

else:
    # 4. Group by product_id and aggregate predicted values
    # The predicted column name includes the period
    predicted_col = f'{TARGET}_predicted_{PREDICTION_PERIOD}'

    if predicted_col not in df_filtered.columns:
        print(f"Error: Predicted column '{predicted_col}' not found in the filtered data.")
        print(f"Available columns: {df_filtered.columns.tolist()}")
        exit()
    # Remove check for actual_col: if actual_col not in df_filtered.columns: ...

    # Aggregate predicted values
    aggregated_predictions_df = df_filtered.groupby('product_id')[predicted_col].sum().reset_index()
    aggregated_predictions_df.rename(columns={predicted_col: 'total_predicted_tn'}, inplace=True)
    print(f"Aggregated predicted data shape: {aggregated_predictions_df.shape}")
    print("Aggregated Prediction Examples:")
    print(aggregated_predictions_df.head())

    # 5. Ensure predicted values are non-negative
    print("\nEnsuring predicted 'total_predicted_tn' values are non-negative...")
    aggregated_predictions_df['total_predicted_tn'] = aggregated_predictions_df['total_predicted_tn'].apply(lambda x: max(0, x))
    print("Negative predicted values set to 0.")


# 6. Save the aggregated predicted results to a CSV file
try:
    # Ensure the output directory exists
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    aggregated_predictions_df.to_csv(AGGREGATED_PREDICTIONS_FILE, index=False)
    print(f"Aggregated predictions saved to {AGGREGATED_PREDICTIONS_FILE}")
    
    # Remove saving aggregated actuals: aggregated_actuals_df.to_csv(...)
    
except Exception as e:
    print(f"Error saving aggregated predictions file: {e}")

# Update the final message
print("Prediction aggregation script finished.")
