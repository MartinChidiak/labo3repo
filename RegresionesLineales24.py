
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os
from datetime import datetime
from sklearn.linear_model import HuberRegressor

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

# --- 1. Armado del Dataset ---
print("Cargando datos de sell-in...")
df_raw = pd.read_csv("sell-in.txt", delimiter='\t')

df_agg = (
    df_raw.groupby(['product_id', 'periodo'], as_index=False)
          .agg({'tn': 'sum'})
          .sort_values(['product_id', 'periodo'])
)

print("Shape del dataset agregado:", df_agg.shape)

# --- 2. Cálculo de la Clase (target) ---
df_agg['periodo'] = df_agg['periodo'].astype(int)
periodos_ordenados = sorted(df_agg['periodo'].unique())
map_periodo_to_mesabs = {p: i + 1 for i, p in enumerate(periodos_ordenados)}
df_agg['mes_abs'] = df_agg['periodo'].map(map_periodo_to_mesabs)
df_agg = df_agg.sort_values(['product_id', 'mes_abs'])
df_agg['tn+2'] = df_agg.groupby('product_id')['tn'].shift(-2)

# --- 3. Feature Engineering: Lags ---
for lag in range(1, 24):
    df_agg[f'tn_{lag}'] = df_agg.groupby('product_id')['tn'].shift(lag)

# --- 4. Dataset de Entrenamiento ---
# Lista de product_id mágicos (puedes ajustar según tu caso)
magicos = [20002, 20003, 20006, 20010, 20011, 20018, 20019, 20021,
   20026, 20028, 20035, 20039, 20042, 20044, 20045, 20046, 20049,
   20051, 20052, 20053, 20055, 20008, 20001, 20017, 20086, 20180,
   20193, 20320, 20532, 20612, 20637, 20807, 20838]


Prods12Meses = [20004,20005,20007,20009,20012,20013,20014,20015,20016,20020,20022,20023,20024,20025,20027,20029,20030,20031,20033,20037,20038,20041,20043,20047,20048,20050,20054,20056,20057,20058,20059,20061,20062,20063,20065,20066,20067,20068,20069,20070,20071,20072,20073,20074,20075,20076,20077,20078,20079,20080,20081,20082,20084,20085,20087,20088,20089,20090,20091,20092,20093,20094,20095,20096,20097,20099,20100,20101,20102,20103,20105,20106,20107,20108,20109,20111,20112,20113,20114,20116,20117,20118,20119,20120,20121,20122,20123,20124,20125,20126,20128,20129,20130,20132,20133,20134,20135,20136,20137,20138,20139,20140,20142,20143,20144,20145,20146,20148,20149,20150,20151,20152,20153,20155,20157,20158,20159,20160,20161,20162,20163,20164,20165,20166,20167,20168,20169,20170,20171,20173,20175,20176,20177,20178,20179,20181,20182,20183,20184,20185,20186,20187,20188,20189,20190,20191,20192,20194,20196,20197,20198,20200,20201,20202,20203,20205,20206,20207,20208,20209,20211,20212,20215,20216,20217,20218,20219,20220,20222,20224,20225,20226,20227,20228,20229,20230,20231,20232,20233,20234,20235,20237,20238,20239,20240,20241,20242,20244,20246,20249,20250,20251,20252,20253,20254,20255,20256,20259,20262,20263,20264,20265,20266,20267,20268,20269,20270,20271,20272,20273,20275,20276,20277,20278,20280,20281,20282,20283,20284,20285,20288,20289,20290,20291,20292,20295,20296,20297,20298,20299,20300,20301,20302,20303,20304,20305,20306,20307,20308,20309,20310,20311,20312,20313,20314,20315,20316,20317,20319,20321,20322,20323,20324,20325,20326,20327,20328,20329,20330,20332,20334,20335,20336,20337,20338,20340,20341,20342,20343,20344,20345,20346,20348,20349,20350,20351,20352,20353,20354,20356,20357,20358,20359,20360,20361,20362,20364,20365,20366,20367,20368,20372,20375,20376,20377,20378,20379,20380,20381,20382,20383,20384,20385,20386,20387,20388,20389,20390,20394,20395,20396,20398,20399,20400,20401,20402,20403,20404,20406,20407,20408,20409,20410,20411,20412,20413,20415,20416,20417,20418,20419,20421,20422,20424,20426,20428,20429,20432,20433,20434,20435,20438,20443,20447,20449,20450,20453,20454,20456,20459,20460,20463,20464,20465,20466,20469,20470,20471,20473,20474,20477,20478,20479,20480,20481,20482,20483,20484,20485,20488,20490,20495,20496,20497,20500,20501,20502,20503,20505,20507,20508,20509,20512,20513,20514,20517,20520,20522,20523,20524,20527,20530,20536,20538,20539,20540,20541,20542,20544,20546,20547,20549,20551,20552,20553,20555,20556,20558,20559,20561,20563,20564,20565,20567,20568,20569,20570,20571,20572,20574,20576,20578,20579,20580,20583,20585,20586,20588,20589,20594,20595,20597,20599,20600,20601,20602,20604,20605,20606,20609,20611,20614,20617,20622,20624,20627,20628,20629,20632,20636,20638,20639,20640,20642,20644,20645,20646,20647,20651,20652,20653,20654,20655,20657,20658,20660,20661,20663,20664,20666,20667,20669,20670,20672,20676,20677,20678,20679,20680,20682,20684,20685,20689,20693,20696,20697,20699,20700,20701,20702,20705,20706,20708,20709,20710,20712,20713,20714,20715,20724,20725,20729,20730,20733,20735,20737,20739,20741,20742,20743,20744,20745,20749,20750,20751,20756,20758,20759,20761,20763,20765,20768,20771,20773,20775,20777,20778,20780,20781,20783,20786,20788,20789,20793,20796,20798,20800,20801,20802,20803,20809,20810,20811,20812,20817,20818,20820,20821,20823,20824,20826,20830,20831,20832,20835,20836,20840,20843,20846,20847,20849,20850,20852,20853,20855,20862,20863,20864,20865,20870,20873,20874,20877,20878,20882,20883,20885,20892,20894,20901,20902,20906,20908,20913,20914,20917,20922,20925,20931,20935,20937,20941,20945,20947,20948,20949,20951,20952,20956,20957,20960,20961,20965,20967,20970,20973,20974,20976,20977,20981,20982,20985,20986,20990,20991,20994,20996,20997,21001,21003,21005,21008,21013,21014,21016,21022,21024,21027,21028,21032,21034,21037,21038,21040,21048,21049,21055,21057,21063,21065,21071,21077,21080,21084,21088,21093,21102,21105,21118,21124,21126,21131,21133,21142,21155,21156,21157,21164,21167,21170,21176,21180,21181,21184,21191,21192,21194,21195,21201,21207,21209,21212,21218,21224,21226,21233,21244,21245,21255,21257,21271]

Prods24Meses = [20004,20005,20007,20009,20012,20013,20014,20015,20016,20020,20022,20023,20024,20025,20027,20029,20030,20031,20033,20037,20038,20041,20043,20047,20048,20050,20054,20056,20057,20058,20059,20061,20062,20063,20065,20066,20067,20068,20069,20070,20071,20072,20073,20074,20075,20076,20077,20078,20079,20080,20081,20082,20084,20087,20088,20090,20091,20092,20093,20094,20095,20096,20097,20099,20100,20101,20102,20103,20105,20106,20107,20108,20109,20111,20112,20113,20114,20116,20117,20118,20119,20120,20121,20122,20123,20124,20125,20128,20129,20132,20133,20134,20137,20138,20139,20140,20142,20144,20145,20146,20148,20149,20151,20152,20153,20155,20157,20158,20160,20161,20162,20163,20164,20165,20166,20167,20168,20169,20171,20173,20175,20176,20177,20178,20179,20181,20182,20183,20184,20185,20187,20188,20189,20190,20191,20194,20196,20197,20198,20200,20201,20202,20203,20205,20206,20207,20208,20209,20211,20212,20215,20216,20217,20218,20219,20220,20222,20224,20225,20226,20227,20228,20230,20231,20232,20233,20234,20235,20238,20239,20240,20241,20242,20244,20246,20249,20250,20251,20252,20253,20254,20255,20256,20259,20263,20264,20265,20267,20268,20269,20270,20271,20272,20273,20275,20276,20277,20278,20280,20281,20282,20283,20284,20285,20288,20289,20290,20291,20292,20295,20296,20297,20298,20299,20300,20301,20302,20303,20304,20305,20306,20307,20308,20309,20310,20311,20312,20313,20314,20315,20316,20317,20319,20321,20322,20323,20324,20325,20326,20327,20328,20329,20330,20332,20334,20335,20336,20337,20338,20340,20341,20342,20343,20344,20346,20348,20349,20350,20352,20353,20354,20356,20357,20358,20359,20360,20361,20362,20365,20366,20367,20372,20375,20376,20377,20379,20380,20381,20382,20383,20384,20385,20386,20387,20388,20390,20394,20396,20398,20399,20400,20401,20402,20403,20404,20406,20407,20409,20410,20411,20412,20413,20415,20416,20417,20418,20419,20421,20422,20424,20426,20428,20429,20432,20433,20434,20435,20438,20443,20447,20449,20450,20453,20454,20456,20459,20460,20463,20464,20465,20466,20469,20470,20471,20473,20474,20478,20479,20480,20482,20483,20484,20485,20490,20496,20497,20500,20501,20502,20505,20507,20508,20509,20512,20514,20517,20524,20530,20536,20538,20539,20542,20544,20549,20551,20552,20555,20561,20563,20564,20565,20567,20568,20570,20572,20574,20578,20579,20583,20585,20586,20588,20589,20594,20595,20597,20599,20600,20601,20602,20605,20606,20609,20614,20617,20622,20624,20628,20629,20632,20636,20639,20640,20642,20644,20645,20646,20647,20651,20652,20653,20654,20655,20657,20658,20660,20661,20663,20664,20667,20669,20670,20672,20676,20677,20678,20680,20684,20685,20693,20696,20697,20699,20701,20702,20705,20706,20708,20710,20713,20714,20715,20724,20725,20729,20730,20733,20735,20737,20739,20741,20742,20743,20744,20745,20749,20750,20751,20756,20758,20759,20761,20765,20768,20771,20773,20775,20777,20778,20780,20781,20786,20788,20789,20793,20796,20800,20801,20802,20803,20809,20810,20812,20818,20820,20821,20823,20826,20830,20831,20832,20840,20843,20846,20847,20849,20850,20855,20862,20863,20864,20865,20870,20873,20874,20877,20878,20882,20883,20885,20892,20894,20901,20906,20913,20914,20922,20925,20931,20935,20937,20941,20945,20947,20948,20949,20951,20952,20956,20957,20960,20961,20965,20967,20970,20973,20974,20976,20977,20982,20985,20986,20991,20994,20996,21003,21005,21008,21014,21016,21024,21027,21028,21032,21038,21048,21055,21057,21071,21077,21080,21088,21118,21124,21131,21155,21156,21167,21170,21181,21184,21194,21195,21212]

Prods24MesesHC = [20007,20009,20012,20013,20014,20015,20016,20020,20022,20024,20025,20027,20029,20030,20031,20038,20041,20043,20050,20056,20057,20062,20063,20065,20066,20067,20068,20069,20070,20071,20072,20073,20074,20076,20082,20087,20088,20091,20092,20097,20099,20102,20103,20109,20112,20113,20114,20117,20124,20128,20129,20137,20138,20144,20148,20149,20151,20160,20162,20163,20164,20165,20166,20168,20171,20178,20183,20185,20190,20191,20196,20197,20201,20202,20203,20205,20206,20209,20217,20218,20219,20222,20233,20239,20246,20253,20254,20280,20281,20288,20304,20308,20311,20312,20313,20319,20332,20341,20357,20358,20361,20366,20376,20388,20390,20412,20413,20415,20421,20447,20473,20478,20479,20485,20507,20508,20524,20530,20564,20588,20595,20652,20653,20657,20705,20724,20733,20737,20741,20750,20780,20803,20855,20877,20941,20996,21003,21048,21155,21167,21184,21195,21212]

PocosdePrueba = [20007,]

Seleccionados = Prods24Meses




features = ['tn'] + [f'tn_{i}' for i in range(1, 24)]
target = 'tn+2'

MES_ABS_TRAIN = 34
MES_ABS_PRED = 36  # Si tu predicción es para el mismo mes_abs, si no, pon el que corresponda

df_train = df_agg[df_agg['mes_abs'] == MES_ABS_TRAIN].copy()
df_train = df_train[df_train['product_id'].isin(Seleccionados)]
print("Antes de dropna:", df_train.shape)
print("NaNs por columna:")
print(df_train[features + [target]].isnull().sum())
print("Ejemplo de filas con NaNs:")
print(df_train[df_train[features + [target]].isnull().any(axis=1)].head())
df_train = df_train.dropna(subset=features + [target])

print(f"Registros para entrenamiento: {df_train.shape[0]}")
print(df_train[features + [target]].head(10))

# --- 5. Entrenamiento del Modelo (Regresión Lineal) ---
X = df_train[features]
y = df_train['tn+2']

# Usamos HuberRegressor para robustez ante outliers y forzamos predicciones >= 0 porque ventas no pueden ser negativas
model = HuberRegressor()
model.fit(X, y)

coef = pd.DataFrame({
    'feature': ['intercept'] + features,
    'coeficiente': [model.intercept_] + list(model.coef_)
})
coef['abs'] = coef['coeficiente'].abs()
print("Coeficientes del modelo:")
print(coef)

# --- 6. Aplicación del Modelo a los 780 productos para 201912 ---
print("Cargando listado de productos a predecir...")
df_780 = pd.read_csv("ListadoIDS.txt", sep=';', header=None)
df_780.columns = ['product_id']
df_780['product_id'] = df_780['product_id'].astype(str)
df_agg['product_id'] = df_agg['product_id'].astype(str)

# Usar features de 201910 para predecir 201912
df_pred = df_agg[
    (df_agg['mes_abs'] == MES_ABS_TRAIN) & 
    (df_agg['product_id'].isin(df_780['product_id']))
].copy()
df_pred['periodo'] = 201912  # Solo para identificar la predicción

# Asegúrate de que todos los productos del listado estén presentes, aunque no tengan datos reales
df_pred = df_780.merge(df_pred, on='product_id', how='left')
df_pred['mes_abs'] = 24  # asegúrate de que la columna esté

# Si no hay datos reales, las features serán NaN, pero igual se predice
df_pred['completos'] = df_pred[features].notnull().all(axis=1)

# SOLO completos que están en seleccionados


#df_completos = df_pred[(df_pred['completos']) & (df_pred['product_id'].isin(Seleccionados))].copy()
df_completos = df_pred[(df_pred['completos'])].copy()

Seleccionados = [str(x) for x in Seleccionados]

df_completos = df_completos[df_completos['product_id'].isin(Seleccionados)]

# El resto van a incompletos (aunque tengan features completos pero no estén en seleccionados)
df_incompletos = df_pred[~((df_pred['completos']) & (df_pred['product_id'].isin(Seleccionados)))].copy()

# --- PREDICCIÓN PARA COMPLETOS ---
if not df_completos.empty:
    df_completos['pred'] = model.predict(df_completos[features])
    df_completos['pred'] = df_completos['pred'].clip(lower=0)  # Fuerza mínimo 0
    df_completos['pred_tipo'] = 'modelo'

# --- Iteraciones de pesos para incompletos ---
pesos_configs = [
    # Variaciones de base y febrero
    {"desc": "base=0.85, feb=2.5", "base": 0.85, "feb_mult": 2.5, "extra_mult": None},
    {"desc": "base=0.90, feb=2.5", "base": 0.90, "feb_mult": 2.5, "extra_mult": None},
    {"desc": "base=0.85, feb=1.0", "base": 0.85, "feb_mult": 1.0, "extra_mult": None},
    {"desc": "base=0.80, feb=3.0", "base": 0.80, "feb_mult": 3.0, "extra_mult": None},
    {"desc": "base=0.95, feb=2.0", "base": 0.95, "feb_mult": 2.0, "extra_mult": None},
    {"desc": "base=0.70, feb=4.0", "base": 0.70, "feb_mult": 4.0, "extra_mult": None},

    # Énfasis en otros meses (ejemplo: enero anterior, lag 11)
    {"desc": "base=0.85, feb=2.5, ene=2.0", "base": 0.85, "feb_mult": 2.5, "extra_mult": {"idx": 11, "mult": 2.0}},
    {"desc": "base=0.85, feb=2.5, lag5=1.5", "base": 0.85, "feb_mult": 2.5, "extra_mult": {"idx": 5, "mult": 1.5}},
    {"desc": "base=0.85, ultimos3=2.0", "base": 0.85, "feb_mult": 1.0, "extra_mult": {"idx": -1, "mult": 2.0}},  # -1 para el último lag

    # Pesos uniformes
    {"desc": "uniforme", "base": 1.0, "feb_mult": 1.0, "extra_mult": None},

    # Énfasis en el mes más reciente (tn)
    {"desc": "base=0.85, actual=3.0", "base": 0.85, "feb_mult": 1.0, "extra_mult": {"idx": 0, "mult": 3.0}},

    # Énfasis en los 3 últimos meses (tn, tn_1, tn_2)
    {"desc": "base=0.85, ultimos3=2.0", "base": 0.85, "feb_mult": 1.0, "extra_mult": None},  # Se ajusta abajo

    # Énfasis en el promedio de todos los lags (simula un promedio simple)
    {"desc": "promedio_simple", "base": 1.0, "feb_mult": 1.0, "extra_mult": None},

    # Énfasis decreciente más fuerte (base más baja)
    {"desc": "base=0.60, feb=2.5", "base": 0.60, "feb_mult": 2.5, "extra_mult": None},

    # Énfasis en el lag medio (por ejemplo, tn_5)
    {"desc": "base=0.85, lag5=3.0", "base": 0.85, "feb_mult": 1.0, "extra_mult": {"idx": 5, "mult": 3.0}},
]

results = []
best_df_final = None
best_score = float('inf')
best_desc = ""
for config in pesos_configs:
    base = config["base"]
    num_features = len(features)
    pesos = np.array([base**i for i in range(num_features)])
    # Aplica multiplicador a febrero si corresponde (lag 10, es decir, tn_10 si existe)
    if num_features > 10:
        pesos[10] *= config["feb_mult"]
    # Si quieres probar otro multiplicador para otro mes, puedes agregarlo aquí
    if config.get("extra_mult") and num_features > config["extra_mult"]["idx"]:
        pesos[config["extra_mult"]["idx"]] *= config["extra_mult"]["mult"]
    # Casos creativos adicionales
    # Énfasis en los 3 últimos meses (tn, tn_1, tn_2)
    if "ultimos3" in config["desc"]:
        for i in range(min(3, num_features)):
            pesos[i] *= 2.0
    # Pesos uniformes (promedio simple)
    if config["desc"] == "uniforme" or config["desc"] == "promedio_simple":
        pesos = np.ones(num_features)
    # Énfasis en el último lag (más antiguo)
    if config.get("extra_mult") and config["extra_mult"]["idx"] == -1:
        pesos[-1] *= config["extra_mult"]["mult"]
    pesos = pesos / pesos.sum()

    # Predicción para incompletos con estos pesos
    df_incompletos_iter = df_incompletos.copy()
    if not df_incompletos_iter.empty:
        X_incompletos = df_incompletos_iter[features].values
        máscara_validos = ~np.isnan(X_incompletos)
        pesos_expandido = np.tile(pesos, (X_incompletos.shape[0], 1))
        pesos_validos = pesos_expandido * máscara_validos
        suma_ponderada = np.nansum(X_incompletos * pesos_validos, axis=1)
        suma_pesos = np.nansum(pesos_validos, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            df_incompletos_iter['pred'] = np.where(
                suma_pesos != 0,
                suma_ponderada / suma_pesos,
                np.nan
            )
        df_incompletos_iter['pred_tipo'] = f'ponderado_{config["desc"]}'
    else:
        df_incompletos_iter['pred'] = np.nan
        df_incompletos_iter['pred_tipo'] = f'ponderado_{config["desc"]}'

    # Unión final para esta iteración
    df_final_iter = pd.concat([df_completos, df_incompletos_iter], axis=0).sort_values('product_id')

    # Métrica custom solo para productos con valor real
    df_eval_iter = df_final_iter[df_final_iter['tn'].notnull()].copy()
    df_eval_iter = df_eval_iter.rename(columns={'tn': 'tn_real_futuro'})

    if not df_eval_iter.empty:
        metric = AbsolutePercentageErrorOnProductTotal(df_eval_iter, product_id_col='product_id', target_col='tn_real_futuro')
        nombre, valor, _ = metric(df_eval_iter['pred'], df_eval_iter['tn_real_futuro'])
        print(f"\nMétrica {nombre} ({config['desc']}): {valor:.2f}%")
        results.append((config['desc'], valor))
        if valor < best_score:
            best_score = valor
            best_desc = config['desc']
            best_df_final = df_final_iter.copy()
    else:
        print(f"\nNo hay productos con valor real para calcular la métrica ({config['desc']}).")

# Al final, muestra el mejor
if results:
    print(f"\nMejor configuración: {best_desc} con score {best_score:.2f}%")
    # Usa el df_final de la mejor configuración para los prints finales
    df_final = best_df_final

    # Separa completos e incompletos según el tipo de predicción en la mejor config
    df_completos_best = df_final[df_final['pred_tipo'] == 'modelo']
    df_incompletos_best = df_final[df_final['pred_tipo'].str.startswith('ponderado_')]

    map_mesabs_to_periodo = {v: k for k, v in map_periodo_to_mesabs.items()}
    periodo_train = map_mesabs_to_periodo[MES_ABS_TRAIN]
    periodo_pred = map_mesabs_to_periodo[MES_ABS_PRED]

    print(f"🔍 Suma de TN a {periodo_train}:")
    print(f"Completos (modelo): {df_completos_best['tn'].sum():,.2f}")
    print(f"Incompletos (promedio historia): {df_incompletos_best['tn'].sum():,.2f}")
    print(f"Total final: {df_completos_best['tn'].sum() + df_incompletos_best['tn'].sum():,.2f}")

    print(f"🔍 Suma de predicciones a {periodo_pred}:")
    print(f"Completos (modelo): {df_completos_best['pred'].sum():,.2f}")
    print(f"Incompletos (promedio historia): {df_incompletos_best['pred'].sum():,.2f}")
    print(f"Total final: {df_final['pred'].sum():,.2f}")

    # Nuevo print: suma real de TN para el periodo de predicción
    suma_real_pred = df_agg[df_agg['periodo'] == periodo_pred]['tn'].sum()
    print(f"🔍 Suma REAL de TN a {periodo_pred}: {suma_real_pred:,.2f}")

    print("\n📦 Desglose:")
    print(f"Total a predecir: {df_final.shape[0]}  (esperado: {df_780.shape[0]})")
    print(f"Completos: {df_completos_best.shape[0]}  |  Incompletos: {df_incompletos_best.shape[0]}")
    print(df_final['pred_tipo'].value_counts())
    print(df_final[['product_id', 'periodo', 'pred', 'pred_tipo']].head())

    print("Productos con valor real en test:", df_eval_iter.shape[0])
    print("Suma real total:", df_eval_iter['tn_real_futuro'].sum())
    print("Suma predicha total:", df_eval_iter['pred'].sum())

# Exportar coeficientes del modelo
# coef = pd.DataFrame({
#     'feature': ['intercept'] + features,
#     'coeficiente': [model.intercept_] + list(model.coef_)
# })
# coef['abs'] = coef['coeficiente'].abs()
# coef = coef.sort_values('abs', ascending=False)
# coef.to_csv('coeficientes_regresion_lineal.csv', index=False)
# print("Coeficientes exportados a coeficientes_regresion_lineal.csv")

# Exportar lags, predicción y real para productos completos en el periodo de test
cols_export = ['product_id', 'periodo', 'pred', 'tn'] + features
df_completos_export = df_completos_best[cols_export].copy()
df_completos_export = df_completos_export.rename(columns={'tn': 'real'})
df_completos_export.to_csv('productos_completos_pred_vs_real.csv', index=False)
print("Predicciones y reales exportados a productos_completos_pred_vs_real.csv")

# # --- Exportar resultados ---
# os.makedirs("kaggle", exist_ok=True)
# timestamp = datetime.now().strftime("%Y%m%d_%H%M")
# filename = f"kaggle/predicciones_201912_{timestamp}.csv"
# df_final[['product_id', 'pred']].to_csv(filename, index=False)
# print(f"Archivo guardado como: {filename}")

# print("df_eval shape:", df_eval.shape)
# print("Ejemplo de df_eval:")
# print(df_eval[['product_id', 'tn_real_futuro', 'pred']].head(10))

# Filtrar para la métrica
df_eval_completos = df_completos_best[df_completos_best['tn'].notnull()].copy().rename(columns={'tn': 'tn_real_futuro'})
df_eval_incompletos = df_incompletos_best[df_incompletos_best['tn'].notnull()].copy().rename(columns={'tn': 'tn_real_futuro'})
df_eval_total = df_final[df_final['tn'].notnull()].copy().rename(columns={'tn': 'tn_real_futuro'})

metric = AbsolutePercentageErrorOnProductTotal

if not df_eval_completos.empty:
    nombre, valor, _ = metric(df_eval_completos, product_id_col='product_id', target_col='tn_real_futuro')(
        df_eval_completos['pred'], df_eval_completos['tn_real_futuro'])
    print(f"Error absoluto porcentual (solo completos/modelo): {valor:.2f}%")

if not df_eval_incompletos.empty:
    nombre, valor, _ = metric(df_eval_incompletos, product_id_col='product_id', target_col='tn_real_futuro')(
        df_eval_incompletos['pred'], df_eval_incompletos['tn_real_futuro'])
    print(f"Error absoluto porcentual (solo incompletos/ponderado): {valor:.2f}%")

if not df_eval_total.empty:
    nombre, valor, _ = metric(df_eval_total, product_id_col='product_id', target_col='tn_real_futuro')(
        df_eval_total['pred'], df_eval_total['tn_real_futuro'])
    print(f"Error absoluto porcentual (total): {valor:.2f}%")

print("Productos completos seleccionados:", df_completos_best['product_id'].nunique())
print("Productos incompletos seleccionados:", df_incompletos_best['product_id'].nunique())

print("Productos con features completos (sin filtro de seleccionados):", df_pred[df_pred[features].notnull().all(axis=1)]['product_id'].nunique())

print("Productos seleccionados con features completos:", df_pred[(df_pred[features].notnull().all(axis=1)) & (df_pred['product_id'].isin(Seleccionados))]['product_id'].nunique())

ejemplo = df_pred[df_pred['product_id'] == Seleccionados[0]]



# --- PREDICCIÓN PARA 201912 (proyectando a 202002) ---

# 1. Prepara el dataset de features para 201912
MES_ABS_PRED_FUTURO = 36  # 201912
PERIODO_PRED_FUTURO = 202002  # El periodo que quieres predecir

df_pred_futuro = df_agg[
    (df_agg['mes_abs'] == MES_ABS_PRED_FUTURO) &
    (df_agg['product_id'].isin(df_780['product_id']))
].copy()
df_pred_futuro['periodo'] = PERIODO_PRED_FUTURO

# Asegúrate de que todos los productos estén presentes
df_pred_futuro = df_780.merge(df_pred_futuro, on='product_id', how='left')
df_pred_futuro['mes_abs'] = MES_ABS_PRED_FUTURO

df_pred_futuro['completos'] = df_pred_futuro[features].notnull().all(axis=1)
df_completos_futuro = df_pred_futuro[df_pred_futuro['completos']].copy()
df_completos_futuro = df_completos_futuro[df_completos_futuro['product_id'].isin(Seleccionados)]

df_incompletos_futuro = df_pred_futuro[~((df_pred_futuro['completos']) & (df_pred_futuro['product_id'].isin(Seleccionados)))].copy()

# 2. Aplica el modelo entrenado a los completos
if not df_completos_futuro.empty:
    df_completos_futuro['pred'] = model.predict(df_completos_futuro[features])
    df_completos_futuro['pred'] = df_completos_futuro['pred'].clip(lower=0)
    df_completos_futuro['pred_tipo'] = 'modelo'

# 3. Aplica el mejor método de pesos a los incompletos
# Usa la configuración de pesos del mejor resultado anterior
config = next(c for c in pesos_configs if c['desc'] == best_desc)
base = config["base"]
num_features = len(features)
pesos = np.array([base**i for i in range(num_features)])
if num_features > 10:
    pesos[10] *= config["feb_mult"]
if config.get("extra_mult") and num_features > config["extra_mult"]["idx"]:
    pesos[config["extra_mult"]["idx"]] *= config["extra_mult"]["mult"]
if "ultimos3" in config["desc"]:
    for i in range(min(3, num_features)):
        pesos[i] *= 2.0
if config["desc"] == "uniforme" or config["desc"] == "promedio_simple":
    pesos = np.ones(num_features)
if config.get("extra_mult") and config["extra_mult"]["idx"] == -1:
    pesos[-1] *= config["extra_mult"]["mult"]
pesos = pesos / pesos.sum()

if not df_incompletos_futuro.empty:
    X_incompletos = df_incompletos_futuro[features].values
    máscara_validos = ~np.isnan(X_incompletos)
    pesos_expandido = np.tile(pesos, (X_incompletos.shape[0], 1))
    pesos_validos = pesos_expandido * máscara_validos
    suma_ponderada = np.nansum(X_incompletos * pesos_validos, axis=1)
    suma_pesos = np.nansum(pesos_validos, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        df_incompletos_futuro['pred'] = np.where(
            suma_pesos != 0,
            suma_ponderada / suma_pesos,
            np.nan
        )
    df_incompletos_futuro['pred_tipo'] = f'ponderado_{config["desc"]}'
else:
    df_incompletos_futuro['pred'] = np.nan
    df_incompletos_futuro['pred_tipo'] = f'ponderado_{config["desc"]}'

# 4. Junta y exporta
df_final_futuro = pd.concat([df_completos_futuro, df_incompletos_futuro], axis=0).sort_values('product_id')

# Exporta a CSV
df_final_futuro[['product_id', 'periodo', 'pred', 'pred_tipo']].to_csv('predicciones_202002.csv', index=False)
print("Predicciones para 202002 exportadas a predicciones_202002.csv")
print(df_final_futuro[['product_id', 'periodo', 'pred', 'pred_tipo']].head())