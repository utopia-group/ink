package streamingRetention.usecases.riot

import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.sql.expressions.Aggregator

case class MhealthInputTupleGL(
  accelerationChestX: Double,
  accelerationChestY: Double,
  accelerationChestZ: Double,
  timestamp: Long,
  stimulus: Long,
  key: String
)

case class StatsAccumulator(
  xAcceleration: List[Double],
  yAcceleration: List[Double],
  zAcceleration: List[Double],
  times: List[Long],
  timestamp: Long,
  stimulus: Long,
  key: String
)

case class Tuple7GL(
  meanX: Double,
  interceptX: Double,
  meanY: Double,
  interceptY: Double,
  meanZ: Double,
  interceptZ: Double,
  key: String,
  timestamp: Long,
  stimulus: Long
)

class StatsMotionLinearRegression extends Aggregator[MhealthInputTupleGL, StatsAccumulator, Tuple7GL] {
  
  def zero: StatsAccumulator = StatsAccumulator(List.empty, List.empty, List.empty, List.empty, -1L, 0L, "")
  
  def reduce(acc: StatsAccumulator, tuple: MhealthInputTupleGL): StatsAccumulator = 
    StatsAccumulator(
      xAcceleration = acc.xAcceleration :+ tuple.accelerationChestX,
      yAcceleration = acc.yAcceleration :+ tuple.accelerationChestY,
      zAcceleration = acc.zAcceleration :+ tuple.accelerationChestZ,
      times = acc.times :+ tuple.timestamp,
      timestamp = if (acc.timestamp > tuple.timestamp) acc.timestamp else tuple.timestamp,
      stimulus = if (acc.stimulus > tuple.stimulus) acc.stimulus else tuple.stimulus,
      key = tuple.key
    )
  
  def finish(acc: StatsAccumulator): Tuple7GL = {
    val meanTimes = acc.times.map(_.toDouble).sum / acc.times.length
    val meanX = acc.xAcceleration.sum / acc.xAcceleration.length
    val meanY = acc.yAcceleration.sum / acc.yAcceleration.length
    val meanZ = acc.zAcceleration.sum / acc.zAcceleration.length
    
    val slopeX = calculateSlope(acc.times, acc.xAcceleration, meanTimes, meanX)
    val slopeY = calculateSlope(acc.times, acc.yAcceleration, meanTimes, meanY)
    val slopeZ = calculateSlope(acc.times, acc.zAcceleration, meanTimes, meanZ)
    
    val interceptX = meanX - slopeX * meanTimes
    val interceptY = meanY - slopeY * meanTimes
    val interceptZ = meanZ - slopeZ * meanTimes
    
    Tuple7GL(meanX, interceptX, meanY, interceptY, meanZ, interceptZ, acc.key, acc.timestamp, acc.stimulus)
  }
  
  private def calculateSlope(x: List[Long], y: List[Double], meanX: Double, meanY: Double): Double = {
    val numerator = x.zip(y).map { case (xi, yi) => (xi - meanX) * (yi - meanY) }.sum
    val denominator = x.map(xi => (xi - meanX) * (xi - meanX)).sum
    numerator / denominator
  }
  
  def bufferEncoder: Encoder[StatsAccumulator] = Encoders.product
  def outputEncoder: Encoder[Tuple7GL] = Encoders.product
} 