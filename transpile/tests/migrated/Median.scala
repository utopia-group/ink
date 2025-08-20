package WinOps

import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.sql.expressions.Aggregator

case class MedianInput(input: Double)
case class MedianBuffer(values: List[Double])

object Median extends Aggregator[MedianInput, MedianBuffer, Double] {
  
  def zero: MedianBuffer = MedianBuffer(List.empty[Double])
  
  def reduce(buffer: MedianBuffer, input: MedianInput): MedianBuffer = {
    MedianBuffer(input.input :: buffer.values)
  }
  
  def merge(b1: MedianBuffer, b2: MedianBuffer): MedianBuffer = {
    throw new UnsupportedOperationException("merge not implemented")
  }
  
  def finish(reduction: MedianBuffer): Double = {
    val sorted = reduction.values.sorted
    val size = sorted.length
    val middleIndex = size / 2
    
    if (size == 0) {
      0.0
    } else if (size % 2 == 0) {
      (sorted(middleIndex - 1) + sorted(middleIndex)) / 2
    } else {
      sorted(middleIndex)
    }
  }
  
  def bufferEncoder: Encoder[MedianBuffer] = Encoders.product
  def outputEncoder: Encoder[Double] = Encoders.DOUBLE
} 