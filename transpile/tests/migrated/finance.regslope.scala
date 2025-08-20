package metrics

import java.time._
import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.sql.expressions.Aggregator

case class RegSlopeInput(year: Long, production: Long)
case class RegSlopeBuffer(sumx: Double, sumy: Double, sumxy: Double, sumx2: Double, count: Long)

class RegSlope extends Aggregator[RegSlopeInput, RegSlopeBuffer, Double] {
  
  val t0 = LocalDate.of(2000, 1, 1)
  
  def xtime(y: Long, m: Long): Long = {
    val t = LocalDate.of(y.toInt, m.toInt, 1)
    val timecoord = t0.until(t)
    timecoord.toTotalMonths
  }
  
  def zero: RegSlopeBuffer = RegSlopeBuffer(0.0, 0.0, 0.0, 0.0, 0L)
  
  def reduce(buffer: RegSlopeBuffer, input: RegSlopeInput): RegSlopeBuffer = {
    val x = ((input.year - 2000) * 12) + 1
    val p = input.production
    RegSlopeBuffer(
      sumx = buffer.sumx + x,
      sumy = buffer.sumy + p,
      sumxy = buffer.sumxy + (x * p),
      sumx2 = buffer.sumx2 + (x * x),
      count = buffer.count + 1
    )
  }
  
  def finish(buffer: RegSlopeBuffer): Double = 
    (buffer.count * buffer.sumxy - (buffer.sumx * buffer.sumy)) / 
    (buffer.count * buffer.sumx2 - (buffer.sumx * buffer.sumx))
  
  def bufferEncoder: Encoder[RegSlopeBuffer] = Encoders.product
  def outputEncoder: Encoder[Double] = Encoders.scalaDouble
} 