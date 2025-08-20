package com.flink.example.stream.window.function

import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.sql.expressions.Aggregator

case class TemperatureInput(id: String, temperature: Int)
case class TemperatureAccumulator(key: String, sum: Int, count: Int)

class AvgTemperatureAggregateFunction extends Aggregator[TemperatureInput, TemperatureAccumulator, (String, Double)] {
  
  def zero: TemperatureAccumulator = TemperatureAccumulator("", 0, 0)
  
  def reduce(buffer: TemperatureAccumulator, value: TemperatureInput): TemperatureAccumulator = {
    TemperatureAccumulator(value.id, buffer.sum + value.temperature, buffer.count + 1)
  }
  
  def merge(b1: TemperatureAccumulator, b2: TemperatureAccumulator): TemperatureAccumulator = {
    throw new UnsupportedOperationException("merge not implemented")
  }
  
  def finish(reduction: TemperatureAccumulator): (String, Double) = {
    val avgTemperature = reduction.sum.toDouble / reduction.count
    (reduction.key, avgTemperature)
  }
  
  def bufferEncoder: Encoder[TemperatureAccumulator] = Encoders.product
  def outputEncoder: Encoder[(String, Double)] = Encoders.tuple(Encoders.STRING, Encoders.DOUBLE)
} 