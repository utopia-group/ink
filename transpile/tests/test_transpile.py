from pathlib import Path

import pytest

import transpile
import transpile.spark_aggregator

TEST_FILE_DIR = Path(__file__).parent / "migrated"
INK_FILE_DIR = Path(__file__).parent / "migrated-ink"


@pytest.mark.parametrize(
    "testname",
    [
        "AreaClickUDAF",
        "AverageAggregate",
        "AvgTemperatureAggregateFunction",
        "AVGVSaggregateGL",
        "BloomFilterAggregationFunction",
        "CarCloudCountGL",
        "CityRatioUDAF",
        "ClickstreamAggregate",
        "CountAggregateFunction",
        "counter",
        "Ex_7_UDAF",
        "finance.OrderInputPriceAggregateFunction",
        "finance.Prediction",
        "finance.regslope",
        "finance.StreamingStockTicker",
        "finance.TransactionSummaryAggregator",
        "HarmonicMeanUDAF",
        "HourlyAvg",
        "KeepRowWithMaxAge",
        "LinearRoadAccidentAggregate",
        "logging.DurationAvg",
        "Median",
        "OutliersOnOutliersDetectAggregateFunction",
        "PointAttributionScalaUdaf",
        "QW",
        "sales.UDAF",
        "sales.UDAF2",
        "SmaAggregation",
        "StatsAggregateFunction",
        "StatsMotionLinearRegression",
        "telemetry.AggMapFirst",
        "TelemetryProcessor",
        "UserSessionAggregates",
        "VehicleStatisticsAggregator",
        "WeightedCentroid",
        # "CombineMaps",  # require axioms about merge
        # "EnsembleByKey",  # requires data structure mapping
        # "finance.OHLCAggregator", # complicated types
        # "finance.PaymentsAggregateFunction",  # require complicated case statements
        # "GroupConcatDistinctUDAF",
        # "HyperLogLogAggregationFunction",  # requires data structure mapping
        # "learning.MinPooling",
        # "LocalStatsAggregate",
        # "RasterFunction",
        # "telemetry.AggRowFirst",
        # "telemetry.CollectList",
        # "TermFrequencyAccumulator", 
        # "TopUdaf",
        # "UnionSketchUDAF",  # requires data structure mapping
        # "VectorSumUDAF",  # requires data structure mapping
    ],
)
def test_transpile(testname):
    with open(TEST_FILE_DIR / f"{testname}.scala", "r") as f:
        code = f.read()

    soln_path = INK_FILE_DIR / f"{testname}.ink"
    if not soln_path.exists():
        assert False, f"Solution file {soln_path} does not exist. Please add it."

    with open(INK_FILE_DIR / f"{testname}.ink", "r") as f:
        # first line is the expected acc
        # second line is the expected init
        lines = f.readlines()
        expected = lines[0].strip()
        expected_init = lines[1].strip() if len(lines) > 1 else ""

    soln, init = transpile.spark_aggregator.transpile(code)
    assert str(soln) == expected
    assert str(init) == expected_init
