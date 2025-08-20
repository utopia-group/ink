package io.palyvos.provenance.usecases.smartgrid.provenance

import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.sql.expressions.Aggregator

case class SmartGridTupleGL(timestamp: Long, stimulus: Long)
case class CountTupleGL(timestamp: Long, key: String, stimulus: Long, count: Long)
case class CountAccumulator(count: Long, key: String, timestamp: Long, stimulus: Long)

class CountAggregateFunction extends Aggregator[SmartGridTupleGL, CountAccumulator, CountTupleGL] {
  
  def zero: CountAccumulator = CountAccumulator(0L, "", Long.MinValue, Long.MinValue)
  
  def reduce(buffer: CountAccumulator, value: SmartGridTupleGL): CountAccumulator = {
    val newTimestamp = if (value.timestamp > buffer.timestamp) value.timestamp else buffer.timestamp
    val newStimulus = if (value.stimulus > buffer.stimulus) value.stimulus else buffer.stimulus
    CountAccumulator(buffer.count + 1, buffer.key, newTimestamp, newStimulus)
  }
  
  def merge(b1: CountAccumulator, b2: CountAccumulator): CountAccumulator = {
    throw new UnsupportedOperationException("merge not implemented")
  }
  
  def finish(reduction: CountAccumulator): CountTupleGL = {
    CountTupleGL(reduction.timestamp, reduction.key, reduction.stimulus, reduction.count)
  }
  
  def bufferEncoder: Encoder[CountAccumulator] = Encoders.product
  def outputEncoder: Encoder[CountTupleGL] = Encoders.product
} 