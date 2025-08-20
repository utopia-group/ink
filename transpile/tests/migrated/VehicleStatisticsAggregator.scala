package com.helecloud.streams.demo.processing

import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.sql.expressions.Aggregator

case class Vehicle(vehicleType: String, timestamp: Long)
case class VehicleStatistics(vehicleType: String, start: Long, end: Long, count: Int)

class VehicleStatisticsAggregator extends Aggregator[Vehicle, VehicleStatistics, VehicleStatistics] {
  
  def zero: VehicleStatistics = VehicleStatistics("", 0L, 0L, 0)
  
  def reduce(buffer: VehicleStatistics, vehicle: Vehicle): VehicleStatistics = {
    val vehicleType = if (buffer.vehicleType == "") vehicle.vehicleType else buffer.vehicleType
    val start = if (buffer.start == 0L) vehicle.timestamp else buffer.start
    val count = if (buffer.count == 0) 1 else buffer.count + 1
    VehicleStatistics(vehicleType, start, vehicle.timestamp, count)
  }
  
  def merge(b1: VehicleStatistics, b2: VehicleStatistics): VehicleStatistics = {
    throw new UnsupportedOperationException("merge not implemented")
  }
  
  def finish(reduction: VehicleStatistics): VehicleStatistics = reduction
  
  def bufferEncoder: Encoder[VehicleStatistics] = Encoders.product
  def outputEncoder: Encoder[VehicleStatistics] = Encoders.product
} 