package io.palyvos.provenance.usecases.cars.cloud.provenance

import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.sql.expressions.Aggregator

case class CarCloudInput(f0: Int, f1: Double, f2: Double, f3: Long, timestamp: Long)

case class CarCloudAccumulator(f0: Int, f1: Int, f2: Long, timestamp: Long)

case class CarCloudOutput(f0: Int, f1: Int, f2: Long, timestamp: Long, stimulus: Long)

class CarCloudCountGL extends Aggregator[CarCloudInput, CarCloudAccumulator, CarCloudOutput] {

  def zero: CarCloudAccumulator = CarCloudAccumulator(0, 0, Long.MinValue, Long.MinValue)

  def reduce(acc: CarCloudAccumulator, input: CarCloudInput): CarCloudAccumulator = {
    val newTimestamp = if (input.timestamp > acc.timestamp) input.timestamp else acc.timestamp
    CarCloudAccumulator(
      f0 = input.f0,
      f1 = acc.f1 + 1,
      f2 = if (input.f3 > acc.f2) input.f3 else acc.f2,
      timestamp = newTimestamp
    )
  }

  def finish(acc: CarCloudAccumulator): CarCloudOutput = {
    CarCloudOutput(
      f0 = acc.f0,
      f1 = acc.f1,
      f2 = acc.f2,
      timestamp = acc.timestamp,
      stimulus = acc.f2
    )
  }

  def bufferEncoder: Encoder[CarCloudAccumulator] = Encoders.product
  def outputEncoder: Encoder[CarCloudOutput] = Encoders.product
}