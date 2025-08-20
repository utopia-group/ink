package barley.analytics.spark.core.udaf.defaults

import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.sql.expressions.Aggregator

case class ValueInput(value: Double)
case class HarmonicBuffer(count: Long, harmonicSum: Double, zeroFound: Boolean)

class HarmonicMeanUDAF extends Aggregator[ValueInput, HarmonicBuffer, Double] {
  
  def zero: HarmonicBuffer = HarmonicBuffer(0L, 0.0, zeroFound = false)
  
  def reduce(buffer: HarmonicBuffer, input: ValueInput): HarmonicBuffer = {
    if (input.value == 0.0 || buffer.zeroFound) {
      HarmonicBuffer(buffer.count, buffer.harmonicSum, true)
    } else {
      HarmonicBuffer(buffer.count + 1, buffer.harmonicSum + (1 / input.value), buffer.zeroFound)
    }
  }
  
  def merge(b1: HarmonicBuffer, b2: HarmonicBuffer): HarmonicBuffer = {
    throw new UnsupportedOperationException("merge not implemented")
  }
  
  def finish(reduction: HarmonicBuffer): Double = {
    if (reduction.zeroFound) {
      0.0
    } else {
      reduction.count / reduction.harmonicSum
    }
  }
  
  def bufferEncoder: Encoder[HarmonicBuffer] = Encoders.product
  def outputEncoder: Encoder[Double] = Encoders.DOUBLE
  
  private def inv(value: Double): Double = {
    1.0 / value
  }
} 