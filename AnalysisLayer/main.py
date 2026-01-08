# %%
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *

# %%
spark = (SparkSession.builder
    .appName("air-traffic-year-join-ml")
    .master("local[*]")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)

spark.version

# %%
TRAFFIC_CSV_PATH = "../DataStorageLayer/export/ExportVerkehr.csv"
AIR_CSV_PATH     = "../DataStorageLayer/export/ExportSchadstoff.csv"

CSV_SEP = ";"
HAS_HEADER = True

# %%
traffic_raw = (spark.read
               .option("header", str(HAS_HEADER).lower())
               .option("sep", CSV_SEP)
               .option("inferSchema", "false")
               .csv(TRAFFIC_CSV_PATH))

air_raw = (spark.read
           .option("header", str(HAS_HEADER).lower())
           .option("sep", CSV_SEP)
           .option("inferSchema", "false")
           .csv(AIR_CSV_PATH))

print("traffic rows:", traffic_raw.count(), "cols:", len(traffic_raw.columns))
print("air rows:", air_raw.count(), "cols:", len(air_raw.columns))

traffic_raw.show(5, truncate=False)
air_raw.show(5, truncate=False)

# %%
traffic_raw = traffic_raw.toDF(*[c.strip() for c in traffic_raw.columns])
air_raw = air_raw.toDF(*[c.strip() for c in air_raw.columns])

print("Traffic columns:", traffic_raw.columns)
print("Air columns:", air_raw.columns)

# %%
def cast_de_number_safe(df, colname: str):
    """
    Robust: ' 215.855,04 ' -> 215855.04 (double)
    ' NA ' / '' / '-' -> NULL
    Ungültige Werte -> NULL (durch SQL try_cast)
    """
    tmp = f"__{colname}_norm"

    # 1) trim + string
    df = df.withColumn(tmp, F.trim(F.col(colname).cast("string")))

    # 2) NA/Noise -> NULL
    df = df.withColumn(
        tmp,
        F.when(
            F.col(tmp).isNull()
            | (F.col(tmp) == "")
            | (F.lower(F.col(tmp)).isin("na", "n/a", "null", "none", "-", "—")),
            F.lit(None),
        ).otherwise(F.col(tmp))
    )

    # 3) de-DE cleanup
    df = df.withColumn(tmp, F.regexp_replace(F.col(tmp), r"\.", ""))  # Tausenderpunkte raus
    df = df.withColumn(tmp, F.regexp_replace(F.col(tmp), r",", "."))  # Komma -> Punkt

    # 4) try_cast (SQL) -> ungültig wird NULL statt Fehler
    df = df.withColumn(colname, F.expr(f"try_cast({tmp} as double)")).drop(tmp)

    return df

# %%
# Verkehr.csv
TRAFFIC_YEAR_COL  = "YEAR"
TRAFFIC_VALUE_COL = "ROAD_TRAFFIC"

# Schadstoff.csv
AIR_REGION_COL    = "Region"
AIR_YEAR_COL      = "Jahr"
AIR_POLLUTANT_COL = "Schadstoff"
AIR_VALUE_COL     = "Werte"   # nach trimmen heißt es "Werte"

TARGET_REGION = "Wien"

# %%
traffic = traffic_raw
# YEAR kann auch mal Spaces haben -> trimmen + cast
traffic = traffic.withColumn(TRAFFIC_YEAR_COL, F.trim(F.col(TRAFFIC_YEAR_COL)).cast("int"))
 
# ROAD_TRAFFIC safe zu double
traffic = cast_de_number_safe(traffic, TRAFFIC_VALUE_COL)

# Verkehr ist laut dir ohnehin Wien -> Region-Spalte setzen
traffic = traffic.withColumn("Region", F.lit(TARGET_REGION))

traffic.select(TRAFFIC_YEAR_COL, "Region", TRAFFIC_VALUE_COL).show(10, truncate=False)

# %%
air = air_raw

air = air.withColumn(AIR_YEAR_COL, F.col(AIR_YEAR_COL).cast("int"))
air = air.filter(F.trim(F.col(AIR_REGION_COL)) == TARGET_REGION)

air = cast_de_number_safe(air, AIR_VALUE_COL)

air.select(AIR_REGION_COL, AIR_YEAR_COL, AIR_POLLUTANT_COL, AIR_VALUE_COL).show(10, truncate=False)

# %%
# Verkehr pro Jahr (falls mehrere Zeilen pro Jahr vorhanden sind: avg)
traffic_year = (traffic
    .groupBy(TRAFFIC_YEAR_COL, "Region")
    .agg(F.avg(F.col(TRAFFIC_VALUE_COL)).alias("traffic_road_traffic_avg"))
)

traffic_year.orderBy(TRAFFIC_YEAR_COL).show(30, truncate=False)

# %%
# Schadstoffe pro Jahr & Schadstoff (avg)
air_year = (air
    .groupBy(AIR_YEAR_COL, AIR_POLLUTANT_COL)
    .agg(F.avg(F.col(AIR_VALUE_COL)).alias("poll_value_avg"))
)

air_year.orderBy(AIR_YEAR_COL, AIR_POLLUTANT_COL).show(30, truncate=False)

# %%
air_year_pivot = (air_year
    .groupBy(AIR_YEAR_COL)
    .pivot(AIR_POLLUTANT_COL)
    .agg(F.first("poll_value_avg"))
)

air_year_pivot.orderBy(AIR_YEAR_COL).show(30, truncate=False)
print("Pivot columns:", air_year_pivot.columns)

# %%
joined = (air_year_pivot
    .join(
        traffic_year.withColumnRenamed(TRAFFIC_YEAR_COL, "YEAR_join"),
        air_year_pivot[AIR_YEAR_COL] == F.col("YEAR_join"),
        how="left"
    )
    .drop("YEAR_join")
)

joined.orderBy(AIR_YEAR_COL).show(50, truncate=False)

# %%
# Jahre in Schadstoffen (Wien) ohne Verkehrseintrag
missing_traffic_years = (joined
    .filter(F.col("traffic_road_traffic_avg").isNull())
    .select(AIR_YEAR_COL)
    .orderBy(AIR_YEAR_COL))

missing_traffic_years.show(200, truncate=False)

# %%
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

import os
import pandas as pd
import matplotlib.pyplot as plt


# %%
# ---------------------------
# HIER ANPASSEN
# ---------------------------

# Beispiel: setze hier einen existierenden Schadstoff-Spaltennamen aus dem Pivot:
# LABEL_COL = # oder "SO2", "NMVOC", "NH3", "PM2_5"
pollutants = ["NOX", "SO2", "NMVOC", "NH3", "PM2_5"]

# Features: fürs Erste nur der Traffic (du kannst später weitere Features ergänzen)
FEATURE_COLS = ["traffic_road_traffic_avg", "year_feature"]

# Option 1: fehlende Traffic-Werte auf 0 setzen (nur fürs schnelle Testen!)
IMPUTE_MISSING_TRAFFIC_WITH_ZERO = True

PLOTS_DIR = "../DataOutputLayer/"

# %%
if IMPUTE_MISSING_TRAFFIC_WITH_ZERO:
    model_df = joined.fillna({"traffic_road_traffic_avg": 0.0})
else:
    model_df = joined

# Wenn du schon joined hast:
if "PM2.5" in joined.columns:
    joined = joined.withColumnRenamed("PM2.5", "PM2_5")

model_df = model_df.withColumn("year_feature", F.col("Jahr").cast("double"))

print("Spalten im joined:", model_df.columns)

# %%

results = []

for label in pollutants:
    data = model_df.select("Jahr", label, *FEATURE_COLS).dropna(subset=[label] + FEATURE_COLS)
    
    if data.count() < 10:
        print(f"{label}: zu wenig Daten")
        continue
    
    train, test = data.randomSplit([0.8, 0.2], seed=42)

    assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features")
    model = RandomForestRegressor(
        featuresCol="features",
        labelCol=label,
        numTrees=200,
        maxDepth=8,
        seed=42
    )

    pipe = Pipeline(stages=[assembler, model])
    fitted = pipe.fit(train)
    preds = fitted.transform(test)

    rmse = RegressionEvaluator(labelCol=label, predictionCol="prediction", metricName="rmse").evaluate(preds)
    r2 = RegressionEvaluator(labelCol=label, predictionCol="prediction", metricName="r2").evaluate(preds)

    results.append((label, rmse, r2))
    print(f"{label}: RMSE={rmse:.2f}, R2={r2:.3f}")
    
    # ---- Plot-Daten als pandas (wenige Jahre → ok) ----
    pdf = preds.toPandas().sort_values("Jahr")

    # 1) Zeitreihe: Ist vs Prognose
    plt.figure()
    plt.plot(pdf["Jahr"], pdf[label], marker="o")
    plt.plot(pdf["Jahr"], pdf["prediction"], marker="o")
    plt.xlabel("Jahr")
    plt.ylabel(label)
    plt.title(f"{label}: Ist vs Prognose (Test) | R2={r2:.3f}, RMSE={rmse:.2f}")
    plt.legend(["Ist", "Prognose"])
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{label}_timeseries_test.png"), dpi=200)
    plt.close()

    # 2) Scatter: Ist vs Prognose + 45° Linie
    plt.figure()
    plt.scatter(pdf[label], pdf["prediction"])
    mn = min(pdf[label].min(), pdf["prediction"].min())
    mx = max(pdf[label].max(), pdf["prediction"].max())
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("Ist")
    plt.ylabel("Prognose")
    plt.title(f"{label}: Ist vs Prognose (Scatter, Test) | R2={r2:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{label}_scatter_test.png"), dpi=200)
    plt.close()



# %%

if results:
    res_df = pd.DataFrame(results, columns=["pollutant", "rmse", "r2"]).sort_values("pollutant")
    res_df.to_csv(os.path.join(PLOTS_DIR, "metrics_overview.csv"), index=False)

    # R2 Balken
    plt.figure()
    plt.bar(res_df["pollutant"], res_df["r2"])
    plt.xlabel("Schadstoff")
    plt.ylabel("R2")
    plt.title("Modellgüte (R2) pro Schadstoff")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "overview_r2.png"), dpi=200)
    plt.close()

    # RMSE Balken
    plt.figure()
    plt.bar(res_df["pollutant"], res_df["rmse"])
    plt.xlabel("Schadstoff")
    plt.ylabel("RMSE")
    plt.title("Fehler (RMSE) pro Schadstoff")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "overview_rmse.png"), dpi=200)
    plt.close()

    print("Plots gespeichert unter:", os.path.abspath(PLOTS_DIR))
else:
    print("Keine Ergebnisse zum Plotten erzeugt.")


