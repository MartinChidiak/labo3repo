import pandas as pd
import numpy as np
import os
import time
import pickle
import hashlib
from pathlib import Path
from typing import Optional
from datetime import datetime




from A_funciones_pipeline_v13 import (
    add_purchase_count,
    add_client_product_relationship_age,
    add_purchase_frequency,
    add_seasonality_features,
    add_entropy_feature_by_category,
    filter_products_by_id,
    filter_top_products_by_volume,
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
    calculate_delta_lags,
    add_targeted_standardized_features,
    add_technical_analysis_features
)

# Paths para correr en GCP
# GCS_BUCKET_PATH = '/home/chidiakmartin/gcs-bucket'
# CHECKPOINTS_DIR = os.path.join(GCS_BUCKET_PATH, 'checkpoints5')
# os.makedirs(CHECKPOINTS_DIR, exist_ok=True)


# Paths para correr localmente
GCS_BUCKET_PATH = r"C:\Users\Martin\OneDrive\Maestría\20- Laboratorio3\RepositorioLabo3"
CHECKPOINTS_DIR = os.path.join(GCS_BUCKET_PATH, 'checkpoints')
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# Constantes temporales
TARGET = 'tn'
TARGET_SHIFT = 2
FUTURE_TARGET = f'{TARGET}_future'
SPLIT_DATES = {
    'train_end': '2019-08-01',
    'val_date': '2019-09-01',
    'test_date': '2019-10-01',
    'predict_date': '2019-12-01',
    'max_train_date': '2019-10-01'
}

class PipelineStep:
    """Clase base para todos los pasos del pipeline"""
    def execute(self, pipeline):
        raise NotImplementedError

    def save_artifact(self, pipeline, name, data):
        pipeline.add_artifact(name, data)

class Pipeline:
    def __init__(self):
        self.artifacts = {}
        self.steps = []
        self.use_disk = True
        self.current_step = 0
        self.checkpoint_file = os.path.join(CHECKPOINTS_DIR, 'pipeline_state.pkl')
        os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
        
        # Cargar estado previo si existe
        if os.path.exists(self.checkpoint_file):
            self.load_state()
    
    def save_state(self):
        """Guarda el estado actual del pipeline"""
        state = {
            'current_step': self.current_step,
            'artifacts_keys': list(self.artifacts.keys()),
            'steps_hash': [self._get_step_hash(step) for step in self.steps]
        }
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(state, f)
    
    def load_state(self):
        """Carga el estado guardado del pipeline"""
        with open(self.checkpoint_file, 'rb') as f:
            state = pickle.load(f)
        self.current_step = state['current_step']
        print(f"\nPipeline cargado. Continuando desde paso {self.current_step+1}")
        print(f"Artifacts disponibles: {', '.join(state['artifacts_keys'])}")
    
    def _get_step_hash(self, step):
        """Genera un hash único para cada paso para detectar cambios"""
        step_str = f"{step.__class__.__name__}{pickle.dumps(step)}"
        return hashlib.md5(step_str.encode()).hexdigest()
    
    def add_step(self, step: PipelineStep):
        self.steps.append(step)
    
    def add_artifact(self, name: str, data):
        self.artifacts[name] = data
        if self.use_disk:
            path = os.path.join(CHECKPOINTS_DIR, f"{name}")
            
            # Eliminar versiones anteriores si existen
            for ext in ['.parquet', '.pkl', '.npy']:
                if os.path.exists(path + ext):
                    os.remove(path + ext)
            
            # Guardar según el tipo de dato
            if isinstance(data, pd.DataFrame):
                data.to_parquet(path + '.parquet', compression='gzip')
            elif isinstance(data, pd.Series):
                data.to_frame().to_parquet(path + '.parquet', compression='gzip')
            elif isinstance(data, np.ndarray):
                np.save(path + '.npy', data)
            elif isinstance(data, (list, dict, int, float, str)):
                with open(path + '.pkl', 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Guardar checksum para validación
            checksum = hashlib.md5(pickle.dumps(data)).hexdigest()
            with open(path + '.checksum', 'w') as f:
                f.write(checksum)
            
            self.save_state()
    
    def get_artifact(self, name: str):
        if name in self.artifacts:
            return self.artifacts[name]
        elif self.use_disk:
            path = os.path.join(CHECKPOINTS_DIR, f"{name}")
            
            # Verificar integridad de los datos
            if os.path.exists(path + '.checksum'):
                with open(path + '.checksum', 'r') as f:
                    saved_checksum = f.read().strip()
            else:
                saved_checksum = None
            
            # Cargar según extensión existente
            if os.path.exists(path + '.parquet'):
                data = pd.read_parquet(path + '.parquet')
                if data.shape[1] == 1:  # Era una Series
                    data = data.iloc[:, 0]
            elif os.path.exists(path + '.npy'):
                data = np.load(path + '.npy', allow_pickle=True)
            elif os.path.exists(path + '.pkl'):
                with open(path + '.pkl', 'rb') as f:
                    data = pickle.load(f)
            else:
                return None
            
            # Validar checksum si existe
            if saved_checksum:
                current_checksum = hashlib.md5(pickle.dumps(data)).hexdigest()
                if current_checksum != saved_checksum:
                    print(f"Advertencia: Checksum no coincide para {name}. Los datos pueden estar corruptos.")
            
            # Mantener en memoria si es crítico
            if name in ['raw_data', 'df_featured', 'X_train', 'y_train']:
                self.artifacts[name] = data
            
            return data
        return None
    
    def run(self):
        total_steps = len(self.steps)
        print(f"\nIniciando pipeline con {total_steps} pasos")
        
        for step_idx, step in enumerate(self.steps):
            if step_idx < self.current_step:
                print(f"Paso {step_idx+1}/{total_steps} ya completado. Saltando...")
                continue
                
            step_name = step.__class__.__name__
            print(f"\n{'='*50}")
            print(f"Ejecutando paso {step_idx+1}/{total_steps}: {step_name}")
            print(f"{'='*50}")
            
            try:
                start_time = time.time()
                step.execute(self)
                elapsed = time.time() - start_time
                
                self.current_step = step_idx + 1
                self.save_state()
                
                print(f"\nPaso {step_idx+1} ({step_name}) completado en {elapsed:.2f} segundos")
                print(f"Artifacts actuales: {', '.join(self.artifacts.keys())}")
            except Exception as e:
                print(f"\nError en paso {step_idx+1} ({step_name}): {str(e)}")
                print("Pipeline pausado. Puede reanudarse más tarde.")
                self.save_state()
                raise
        
        print("\nPipeline completado exitosamente!")
        self.cleanup()

    def cleanup(self):
        """Limpieza final después de completar todos los pasos"""
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
        print("Checkpoints limpiados. Pipeline finalizado completamente.")

def reset_pipeline():
    """Elimina todos los checkpoints para empezar desde cero"""
    import shutil
    if os.path.exists(CHECKPOINTS_DIR):
        shutil.rmtree(CHECKPOINTS_DIR)
    os.makedirs(CHECKPOINTS_DIR)
    print("\nTodos los checkpoints han sido eliminados. Pipeline puede iniciarse desde cero.")

# Implementaciones concretas de los pasos del pipeline
class DataLoadingStep(PipelineStep):
    def execute(self, pipeline: Pipeline) -> None:
        # Verificar si ya tenemos los datos cargados
        if pipeline.get_artifact("raw_data") is not None:
            print("Datos crudos ya cargados. Saltando paso...")
            return
            
        print("Loading and preprocessing data...")
        
        # Cargar datos
        sell_in = pd.read_csv(os.path.join(GCS_BUCKET_PATH, 'sell-in.txt'), delimiter='\t')
        productos = pd.read_csv(os.path.join(GCS_BUCKET_PATH, 'tb_productos.txt'), delimiter='\t')
        stocks = pd.read_csv(os.path.join(GCS_BUCKET_PATH, 'tb_stocks.txt'), delimiter='\t')
        
        # Combinar datos
        productos = productos.drop_duplicates(subset=['product_id'])
        df = sell_in.merge(stocks, on=['periodo', 'product_id'], how='left')
        df = df.merge(productos, on='product_id', how='left')
        
        # Optimizar tipos de datos
        df = self.optimize_dtypes(df)
        df['fecha'] = pd.to_datetime(df['periodo'].astype(str), format='%Y%m')
        
        self.save_artifact(pipeline, "raw_data", df)

    def optimize_dtypes(self, df):
        # Optimizar números
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Convertir a categorías donde sea posible
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df[col]) < 0.5:
                df[col] = df[col].astype('category')
        
        return df

class FeatureEngineeringStep(PipelineStep):
    def execute(self, pipeline: Pipeline) -> None:
        if pipeline.get_artifact("df_featured") is not None:
            print("Datos con features ya procesados. Saltando paso...")
            return
            
        print("Running feature engineering...")
        df = pipeline.get_artifact("raw_data")

        
        # 1. Filtrado de productos por lista manual
        ProdsConScoreAltoHC = [20001,20002,20006,20007,20008,20009,20010,20011,20012,20013,20014,20015,20016,20017,20018,20020,20021,20022,20024,20025,20026,20027,20028,20029,20030,20031,20032,20034,20035,20036,20038,20039,20040,20041,20042,20043,20045,20049,20050,20051,20053,20055,20056,20057,20060,20062,20063,20064,20065,20066,20067,20068,20069,20070,20071,20072,20073,20074,20076,20082,20083,20085,20087,20088,20089,20091,20092,20097,20098,20099,20102,20103,20104,20109,20110,20112,20113,20114,20115,20117,20124,20126,20127,20128,20129,20135,20138,20141,20147,20148,20149,20150,20151,20156,20165,20168,20172,20174,20195,20210,20236,20247,20260,20261]

        ProdsConScoreAltoPCyFood = [20003,20004,20005,20019,20023,20033,20037,20044,20046,20047,20048,20052,20054,20058,20059,20061,20075,20077,20078,20079,20080,20081,20084,20086,20090,20093,20094,20095,20096,20100,20101,20105,20107,20108,20116,20118,20119,20121,20130,20131,20136,20143,20153,20154,20157,20159,20170,20199,20213,20223]

        ProdsMagicos = [20002, 20003, 20006, 20010, 20011, 20018,20019, 20021,20026, 20028, 20035, 20039, 20042, 20044, 20045,20046, 20049,20051,20052, 20053, 20055, 20008, 20001, 20017, 20180,20193, 20320, 20532, 20612, 20637, 20807, 20838, 20086]

        MergeProds = list(set(ProdsConScoreAltoHC) & set(ProdsConScoreAltoPCyFood) & set(ProdsMagicos))
                          
        UnionProds = list(set(ProdsConScoreAltoHC + ProdsConScoreAltoPCyFood + ProdsMagicos))

        PRODUCT_IDS_TO_KEEP = UnionProds
        df = filter_products_by_id(df, PRODUCT_IDS_TO_KEEP)
        
        # 2. Generar combinaciones temporales
        combinations_df = generar_combinaciones_por_periodo(df)
        df = merge_with_original_data(combinations_df, df)
        
        # 3. Rellenar información faltante
        productos_df = pd.read_csv(os.path.join(GCS_BUCKET_PATH, 'tb_productos.txt'), delimiter='\t')
        df = fill_missing_product_info(df, productos_df=productos_df)
        
        # 4. Calcular target futuro
        df = df.sort_values(['customer_id', 'product_id', 'fecha'])
        df[FUTURE_TARGET] = df.groupby(['customer_id', 'product_id'])[TARGET].shift(-TARGET_SHIFT)
        
        # 5. Aplicar todas las transformaciones de features (con checkpoint)
        df = self.calculate_all_features(df, pipeline)

        self.save_artifact(pipeline, "df_featured", df)
        # Elimina el checkpoint temporal
        if "df_featured_checkpoint" in pipeline.artifacts:
            del pipeline.artifacts["df_featured_checkpoint"]
        checkpoint_path = os.path.join(CHECKPOINTS_DIR, "df_featured_checkpoint.pkl")
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

    def calculate_all_features(self, df, pipeline):
        """Aplica secuencialmente todas las transformaciones de features y guarda solo el último checkpoint"""
        feature_params = {
            "event_file_path": os.path.join(GCS_BUCKET_PATH, 'eventos_macro.txt'),
            "columnas_para_lag": ['cust_request_qty', 'cust_request_tn', 'tn'],
            "num_lags": 36,
            "window_size": 3,
            "rolling_columns": ["tn"],
            "rolling_windows": list(range(1, 37)),
            "rolling_stats": ["mean"],
            "ta_target_column": "tn",
            "rsi_window": 6,
            "bollinger_window": 12,
            "bollinger_std": 1.5,
            "macd_fast": 3,
            "macd_slow": 12,
            "macd_signal": 6,
            "ulcer_window": 8,
            "kama_window": 6,
            "kama_pow1": 2,
            "kama_pow2": 30
        }
        
        feature_functions = [
            # AGREGADOS: estacionalidad y entropía por categoría
            (add_seasonality_features, {}),
            (add_entropy_feature_by_category, {"category_col": "cat1", "prefix": "cat1"}),
            (add_entropy_feature_by_category, {"category_col": "cat2", "prefix": "cat2"}),
            (add_entropy_feature_by_category, {"category_col": "cat3", "prefix": "cat3"}),
            # NUEVAS funciones de relación cliente-producto
            (add_purchase_count, {}),
            (add_client_product_relationship_age, {}),
            (add_purchase_frequency, {}),

            (calculate_brand_loyalty, {}),
            (add_customer_category_avg_tn, {}),
            (calculate_customer_category_count, {}),
            (add_macro_event_flag, {"event_file_path": feature_params["event_file_path"]}),
            (calculate_tn_percentage_change, {}),
            (calculate_months_since_last_purchase, {}),
            (calculate_product_moving_avg, {}),
            (calculate_weighted_tn_sum, {"window_size": feature_params["window_size"]}),
            (calculate_demand_growth_rate_diff, {}),
            (add_total_tn_per_product, {}),
            (generar_lags_por_combinacion, {
                "columnas_para_lag": feature_params["columnas_para_lag"],
                "num_lags": feature_params["num_lags"]
            }),
            (calculate_delta_lags, {
                "base_columns": feature_params["columnas_para_lag"],
                "max_lag": feature_params["num_lags"] - 1
            }),
            (add_rolling_statistics_features, {
                "columns": feature_params["rolling_columns"],
                "windows": feature_params["rolling_windows"],
                "stats": feature_params["rolling_stats"]
            }),
            (add_exponential_moving_average_features, {}),
            (add_difference_features, {}),
            (add_total_category_sales, {}),
            (add_customer_product_total_weights, {}),
            (add_interaction_features, {}),
            (add_targeted_standardized_features, {}),
            (add_technical_analysis_features, {
                "ta_target_column": feature_params["ta_target_column"],
                "rsi_window": feature_params["rsi_window"],
                "bollinger_window": feature_params["bollinger_window"],
                "bollinger_std": feature_params["bollinger_std"],
                "macd_fast": feature_params["macd_fast"],
                "macd_slow": feature_params["macd_slow"],
                "macd_signal": feature_params["macd_signal"],
                "ulcer_window": feature_params["ulcer_window"],
                "kama_window": feature_params["kama_window"],
                "kama_pow1": feature_params["kama_pow1"],
                "kama_pow2": feature_params["kama_pow2"]
            })
        ]

        # Intenta cargar el checkpoint si existe
        checkpoint = pipeline.get_artifact("df_featured_checkpoint")
        start_idx = 0
        if checkpoint is not None and isinstance(checkpoint, dict):
            df = checkpoint["df"]
            start_idx = checkpoint["step_idx"]
            print(f"Recuperando df_featured_checkpoint: retomando desde el paso {start_idx+1}")

        for idx, (func, params) in enumerate(feature_functions):
            if idx < start_idx:
                continue  # Ya procesado
            print(f"Applying {func.__name__} ({idx+1}/{len(feature_functions)})...")
            start_time = time.time()
            df = func(df, **params)
            elapsed = time.time() - start_time
            print(f"Completed in {elapsed:.2f} seconds. Shape: {df.shape}")

            # Antes de guardar el nuevo checkpoint, borra el anterior si existe
            if "df_featured_checkpoint" in pipeline.artifacts:
                del pipeline.artifacts["df_featured_checkpoint"]
            checkpoint_path = os.path.join(CHECKPOINTS_DIR, "df_featured_checkpoint.pkl")
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)

            # Guarda el nuevo checkpoint
            pipeline.add_artifact("df_featured_checkpoint", {"df": df, "step_idx": idx + 1})

        return df

class TemporalSplitStep(PipelineStep):
    def execute(self, pipeline: Pipeline) -> None:
        if all(pipeline.get_artifact(name) is not None 
               for name in ['df_train', 'df_val', 'df_test', 'df_predict']):
            print("Datos ya divididos temporalmente. Saltando paso...")
            return
            
        print("Splitting data temporally...")
        df = pipeline.get_artifact("df_featured")
        
        if not pd.api.types.is_datetime64_any_dtype(df['fecha']):
            df['fecha'] = pd.to_datetime(df['fecha'])
        
        splits = {
            'df_train': df[df['fecha'] <= pd.to_datetime(SPLIT_DATES['train_end'])],
            'df_val': df[df['fecha'] == pd.to_datetime(SPLIT_DATES['val_date'])],
            'df_test': df[df['fecha'] == pd.to_datetime(SPLIT_DATES['test_date'])],
            'df_predict': df[df['fecha'] == pd.to_datetime(SPLIT_DATES['predict_date'])],
            'df_full_train': df[df['fecha'] <= pd.to_datetime(SPLIT_DATES['max_train_date'])]
        }
        
        for name, data in splits.items():
            self.save_artifact(pipeline, name, data)

class TrainTestPrepareStep(PipelineStep):
    def execute(self, pipeline: Pipeline) -> None:
        if pipeline.get_artifact("X_train") is not None:
            print("Conjuntos de entrenamiento ya preparados. Saltando paso...")
            return
            
        print("Preparing train/test sets...")
        
        train = pipeline.get_artifact("df_train")
        val = pipeline.get_artifact("df_val")
        test = pipeline.get_artifact("df_test")
        predict = pipeline.get_artifact("df_predict")
        full_train = pipeline.get_artifact("df_full_train")
        
        non_features = ['fecha', 'periodo', 'customer_id', 'product_id', FUTURE_TARGET]
        features = [col for col in train.columns if col not in non_features]
        
        artifacts = {
            'X_train': train[features],
            'y_train': train[FUTURE_TARGET],
            'X_val': val[features],
            'y_val': val[FUTURE_TARGET],
            'X_test': test[features],
            'y_test': test[[FUTURE_TARGET, 'product_id']],
            'X_predict': predict[features],
            'X_full_train': full_train[features],
            'y_full_train': full_train[FUTURE_TARGET],
            'feature_names': features
        }
        
        for name, data in artifacts.items():
            self.save_artifact(pipeline, name, data)

if __name__ == "__main__":
    # Para reiniciar completamente (opcional)
    # reset_pipeline()
    
    # Crear y ejecutar pipeline
    pipeline = Pipeline()
    
    # Añadir pasos
    pipeline.add_step(DataLoadingStep())
    pipeline.add_step(FeatureEngineeringStep())
    pipeline.add_step(TemporalSplitStep())
    pipeline.add_step(TrainTestPrepareStep())
    
    # Ejecutar pipeline
    pipeline.run()
    
    print("\nPipeline execution completed successfully!")
    print("Available artifacts:")
    for artifact_name in pipeline.artifacts.keys():
        print(f"- {artifact_name}")