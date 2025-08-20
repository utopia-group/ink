package projekat

import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.sql.expressions.Aggregator

case class Bus(busLine: String, speed: Double)
case class BusStats(busLine: String, speedSum: Double, minSpeed: Double, maxSpeed: Double, count: Double)

class AverageAggregate extends Aggregator[Bus, BusStats, (String, Double, Double, Double, Double)] {
  
  def zero: BusStats = BusStats("", 0.0, Double.MaxValue, Double.MinValue, 0.0)
  
  def reduce(buffer: BusStats, bus: Bus): BusStats = {
    val minSpeed = if (bus.speed < buffer.minSpeed) bus.speed else buffer.minSpeed
    val maxSpeed = if (bus.speed > buffer.maxSpeed) bus.speed else buffer.maxSpeed
    BusStats(bus.busLine, buffer.speedSum + bus.speed, minSpeed, maxSpeed, buffer.count + 1)
  }
  
  def merge(b1: BusStats, b2: BusStats): BusStats = {
    throw new UnsupportedOperationException("merge not implemented")
  }
  
  def finish(reduction: BusStats): (String, Double, Double, Double, Double) = {
    (reduction.busLine, reduction.minSpeed, reduction.maxSpeed, reduction.speedSum / reduction.count, reduction.count)
  }
  
  def bufferEncoder: Encoder[BusStats] = Encoders.product
  def outputEncoder: Encoder[(String, Double, Double, Double, Double)] = Encoders.tuple(Encoders.STRING, Encoders.DOUBLE, Encoders.DOUBLE, Encoders.DOUBLE, Encoders.DOUBLE)
} 