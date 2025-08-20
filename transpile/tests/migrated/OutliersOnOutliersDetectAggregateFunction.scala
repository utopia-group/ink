package net.sparkworks.functions

import net.sparkworks.model.{FlaggedOutliersResult, OutliersResult}
import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.sql.expressions.Aggregator

case class OutliersDetectAccumulator(count: Long, outlierCount: Long)

class OutliersOnOutliersDetectAggregateFunction extends Aggregator[Int, OutliersDetectAccumulator, OutliersResult] {
  
  def zero: OutliersDetectAccumulator = OutliersDetectAccumulator(0L, 0L)
  
  def reduce(acc: OutliersDetectAccumulator, value: Int): OutliersDetectAccumulator = {
    val newOutlierCount = if (value > 100) acc.outlierCount + 1 else acc.outlierCount
    OutliersDetectAccumulator(acc.count + 1, newOutlierCount)
  }
  
  def finish(acc: OutliersDetectAccumulator): OutliersResult = {
    val result = new OutliersResult()
    result.setOutliersCount(acc.count)
    result.setOutliersOnOutliersCount(acc.outlierCount)
    result
  }
  
  def bufferEncoder: Encoder[OutliersDetectAccumulator] = Encoders.product
  def outputEncoder: Encoder[OutliersResult] = Encoders.product
} 