package de.hpi.isg.sindy.util

import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.sql.expressions.Aggregator

case class CounterInput(columnId: Int)
case class CounterBuffer(counts: Map[Int, Long])

class LongCounter extends Aggregator[CounterInput, CounterBuffer, Map[Int, Long]] {
  
  def zero: CounterBuffer = CounterBuffer(Map.empty[Int, Long])
  
  def reduce(buffer: CounterBuffer, input: CounterInput): CounterBuffer = {
    CounterBuffer(buffer.counts + (input.columnId -> (buffer.counts.getOrElse(input.columnId, 0L) + 1L)))
  }
  
  def merge(b1: CounterBuffer, b2: CounterBuffer): CounterBuffer = {
    throw new UnsupportedOperationException("merge not implemented")
  }
  
  def finish(reduction: CounterBuffer): Map[Int, Long] = reduction.counts
  
  def bufferEncoder: Encoder[CounterBuffer] = Encoders.product
  def outputEncoder: Encoder[Map[Int, Long]] = Encoders.kryo[Map[Int, Long]]
} 