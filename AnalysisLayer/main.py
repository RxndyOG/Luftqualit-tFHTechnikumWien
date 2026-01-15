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
TRAFFIC_CSV_PATH = "../DataStorageLayer/export/ExportVerkehr2.csv"
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

TRAFFIC_REGION_SRC_COL = "Bundesland"           
TRAFFIC_VALUE_COL      = "ROAD_TRAFFIC"
TRAFFIC_YEAR_COL       = "YEAR"


year_cols = [c for c in traffic_raw.columns if c.isdigit()]
year_cols_sorted = sorted(year_cols, key=lambda x: int(x))

if not year_cols_sorted:
    raise ValueError("Keine Jahres-Spalten (digit columns) in ExportVerkehr2 gefunden.")


stack_args = ", ".join([f"'{y}', `{y}`" for y in year_cols_sorted])
stack_expr = f"stack({len(year_cols_sorted)}, {stack_args}) as ({TRAFFIC_YEAR_COL}, {TRAFFIC_VALUE_COL})"

traffic_long = (traffic_raw
    .select(F.col(TRAFFIC_REGION_SRC_COL).alias("Region_src"), F.expr(stack_expr))
)

traffic_long = traffic_long.withColumn(TRAFFIC_YEAR_COL, F.col(TRAFFIC_YEAR_COL).cast("int"))
traffic_long = cast_de_number_safe(traffic_long, TRAFFIC_VALUE_COL)

REGION_MAP = {
    "Niederoesterreich": "Niederoestereich",
    "Oberoesterreich": "Oberoestereich",
    "OESTERREICH": "AT",
}
traffic_long = traffic_long.withColumn(
    "Region",
    F.coalesce(
        F.create_map([F.lit(x) for kv in REGION_MAP.items() for x in kv]).getItem(F.col("Region_src")),
        F.col("Region_src")
    )
)

traffic_long.select("Region_src", "Region", TRAFFIC_YEAR_COL, TRAFFIC_VALUE_COL).show(20, truncate=False)

# %%
# Aggregate (if duplicates exist)
traffic_year = (traffic_long
    .groupBy(TRAFFIC_YEAR_COL, "Region")
    .agg(F.avg(F.col(TRAFFIC_VALUE_COL)).alias("traffic_road_traffic_avg"))
)

# %%
# ---------------------------
# Air (Schadstoff)
# ---------------------------
AIR_REGION_COL    = "Region"
AIR_YEAR_COL      = "Jahr"
AIR_POLLUTANT_COL = "Schadstoff"
AIR_VALUE_COL     = "Werte"

air = air_raw.withColumn(AIR_YEAR_COL, F.col(AIR_YEAR_COL).cast("int"))
air = cast_de_number_safe(air, AIR_VALUE_COL)

# If PM2.5 exists, rename to spark-safe column name
if "PM2.5" in air.columns:
    air = air.withColumnRenamed("PM2.5", "PM2_5")


PLOTS_DIR = "../DataOutputLayer/"

# %%

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

import os

import pandas as pd
import matplotlib.pyplot as plt

# %%
pollutants = ["NOX", "SO2", "NMVOC", "NH3", "PM2_5"]
FEATURE_COLS = ["traffic_road_traffic_avg", "year_feature"]

IMPUTE_MISSING_TRAFFIC_WITH_ZERO = True

# Regions present in both datasets
regions_air = [r["Region"] for r in air.select(F.col(AIR_REGION_COL).alias("Region")).distinct().collect()]
regions_traffic = [r["Region"] for r in traffic_year.select("Region").distinct().collect()]
regions = sorted(set(regions_air).intersection(set(regions_traffic)))

print("Regions (intersection):", regions)

# %%
all_metrics = []

all_preds = []

for region in regions:
    # Skip national if you want only provinces/cities; comment out if you want AT too.
    # if region == "AT":
    #     continue

    print("\n==============================")
    print("REGION:", region)
    print("==============================")

    # Filter air for region
    air_r = air.filter(F.trim(F.col(AIR_REGION_COL)) == F.lit(region))

    # Air agg + pivot
    air_year = (air_r
        .groupBy(AIR_YEAR_COL, AIR_POLLUTANT_COL)
        .agg(F.avg(F.col(AIR_VALUE_COL)).alias("poll_value_avg"))
    )

    air_year_pivot = (air_year
        .groupBy(AIR_YEAR_COL)
        .pivot(AIR_POLLUTANT_COL)
        .agg(F.first("poll_value_avg"))
    )

    # Region-specific traffic
    traffic_r = traffic_year.filter(F.col("Region") == F.lit(region))

    joined = (air_year_pivot
        .join(
            traffic_r.withColumnRenamed(TRAFFIC_YEAR_COL, "YEAR_join"),
            air_year_pivot[AIR_YEAR_COL] == F.col("YEAR_join"),
            how="left"
        )
        .drop("YEAR_join")
        .withColumn("Region", F.lit(region))
    )

    if IMPUTE_MISSING_TRAFFIC_WITH_ZERO:
        joined = joined.fillna({"traffic_road_traffic_avg": 0.0})

    # year feature
    model_df = joined.withColumn("year_feature", F.col(AIR_YEAR_COL).cast("double"))

    # If PM2.5 column appears (rare), rename it
    if "PM2.5" in model_df.columns:
        model_df = model_df.withColumnRenamed("PM2.5", "PM2_5")

    # Train models for all pollutants available in this region DF
    region_metrics = []
    for label in pollutants:
        if label not in model_df.columns:
            continue

        data = model_df.select(AIR_YEAR_COL, label, *FEATURE_COLS).dropna(subset=[label] + FEATURE_COLS)

        if data.count() < 10:
            print(f"  {label}: zu wenig Daten")
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
        preds = fitted.transform(test).select(AIR_YEAR_COL, label, "prediction").orderBy(AIR_YEAR_COL)

        preds_with_region = preds.withColumn("Region", F.lit(region)).withColumn("pollutant", F.lit(label))
        # und in eine Liste sammeln:
        all_preds.append(preds_with_region)

        rmse = RegressionEvaluator(labelCol=label, predictionCol="prediction", metricName="rmse").evaluate(preds)
        r2 = RegressionEvaluator(labelCol=label, predictionCol="prediction", metricName="r2").evaluate(preds)

        region_metrics.append((region, label, rmse, r2))
        all_metrics.append((region, label, rmse, r2))

        print(f"  {label}: RMSE={rmse:.2f}, R2={r2:.3f}")

        # --- plots ---
        pdf = preds.toPandas().sort_values(AIR_YEAR_COL)

        # timeseries
        plt.figure()
        plt.plot(pdf[AIR_YEAR_COL], pdf[label], marker="o")
        plt.plot(pdf[AIR_YEAR_COL], pdf["prediction"], marker="o")
        plt.xlabel("Jahr")
        plt.ylabel(label)
        plt.title(f"{region} | {label}: Ist vs Prognose (Test) | R2={r2:.3f}, RMSE={rmse:.2f}")
        plt.legend(["Ist", "Prognose"])
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{region}_{label}_timeseries_test.png"), dpi=200)
        plt.close()

        # scatter
        plt.figure()
        plt.scatter(pdf[label], pdf["prediction"])
        mn = min(pdf[label].min(), pdf["prediction"].min())
        mx = max(pdf[label].max(), pdf["prediction"].max())
        plt.plot([mn, mx], [mn, mx])
        plt.xlabel("Ist")
        plt.ylabel("Prognose")
        plt.title(f"{region} | {label}: Scatter (Test) | R2={r2:.3f}")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{region}_{label}_scatter_test.png"), dpi=200)
        plt.close()

    # Per-region overview plots + metrics CSV
    if region_metrics:
        res_df = pd.DataFrame(region_metrics, columns=["region", "pollutant", "rmse", "r2"]).sort_values("pollutant")
        res_df.to_csv(os.path.join(PLOTS_DIR, f"{region}_metrics_overview.csv"), index=False)

        # R2 bar
        plt.figure()
        plt.bar(res_df["pollutant"], res_df["r2"])
        plt.xlabel("Schadstoff")
        plt.ylabel("R2")
        plt.title(f"{region}: Modellgüte (R2) pro Schadstoff")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{region}_overview_r2.png"), dpi=200)
        plt.close()

        # RMSE bar
        plt.figure()
        plt.bar(res_df["pollutant"], res_df["rmse"])
        plt.xlabel("Schadstoff")
        plt.ylabel("RMSE")
        plt.title(f"{region}: Fehler (RMSE) pro Schadstoff")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{region}_overview_rmse.png"), dpi=200)
        plt.close()



# %%
# Global metrics summary
if all_metrics:
    all_df = pd.DataFrame(all_metrics, columns=["region", "pollutant", "rmse", "r2"]).sort_values(["region","pollutant"])
    all_df.to_csv(os.path.join(PLOTS_DIR, "ALL_regions_metrics.csv"), index=False)
    
    # Aggregation pro Schadstoff
    agg = (all_df
        .groupby("pollutant")
        .agg(
            r2_mean=("r2", "mean"),
            r2_median=("r2", "median"),
            r2_std=("r2", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_median=("rmse", "median"),
            rmse_std=("rmse", "std"),
            n_regions=("region", "nunique")
        )
        .reset_index()
        .sort_values("pollutant")
    )
    agg.to_csv(os.path.join(PLOTS_DIR, "ALL_pollutants_agg.csv"), index=False)

    # Plot: Mean R2 pro Schadstoff (mit errorbar = std)
    plt.figure()
    plt.bar(agg["pollutant"], agg["r2_mean"])
    plt.errorbar(agg["pollutant"], agg["r2_mean"], yerr=agg["r2_std"], fmt="none", capsize=4)
    plt.xlabel("Schadstoff")
    plt.ylabel("R2 (Mean ± Std über Regionen)")
    plt.title("Global: Modellgüte über alle Regionen")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "GLOBAL_mean_r2_with_std.png"), dpi=200)
    plt.close()

    # Plot: Mean RMSE pro Schadstoff (mit errorbar = std)
    plt.figure()
    plt.bar(agg["pollutant"], agg["rmse_mean"])
    plt.errorbar(agg["pollutant"], agg["rmse_mean"], yerr=agg["rmse_std"], fmt="none", capsize=4)
    plt.xlabel("Schadstoff")
    plt.ylabel("RMSE (Mean ± Std über Regionen)")
    plt.title("Global: Fehler über alle Regionen")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "GLOBAL_mean_rmse_with_std.png"), dpi=200)
    plt.close()

    print("✅ Global Aggregates/Plots gespeichert:", os.path.abspath(PLOTS_DIR))
else:
    print("⚠️ Keine globalen Metriken vorhanden.")

# %%
if all_metrics:
    pivot_r2 = all_df.pivot(index="region", columns="pollutant", values="r2").sort_index()
    pivot_r2.to_csv(os.path.join(PLOTS_DIR, "GLOBAL_r2_matrix.csv"))

    plt.figure(figsize=(10, 6))
    plt.imshow(pivot_r2.values, aspect="auto")
    plt.xticks(range(len(pivot_r2.columns)), pivot_r2.columns, rotation=45, ha="right")
    plt.yticks(range(len(pivot_r2.index)), pivot_r2.index)
    plt.colorbar(label="R2")
    plt.title("R2 Heatmap: Regionen × Schadstoffe")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "GLOBAL_r2_heatmap.png"), dpi=200)
    plt.close()

# %%
if all_metrics:
    for pol in sorted(all_df["pollutant"].unique()):
        top = all_df[all_df["pollutant"] == pol].sort_values("r2", ascending=False).head(5)
        top.to_csv(os.path.join(PLOTS_DIR, f"TOP5_{pol}_by_r2.csv"), index=False)

        plt.figure()
        plt.bar(top["region"], top["r2"])
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Region")
        plt.ylabel("R2")
        plt.title(f"Top 5 Regionen nach R2 ({pol})")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"TOP5_{pol}_r2.png"), dpi=200)
        plt.close()



