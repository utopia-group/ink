package sql

import org.apache.spark.sql._
import org.apache.spark.sql.expressions.Aggregator

case class SalesInput(sales: Double, state_id: Int)

class ScalaAggregateFunction extends Aggregator[SalesInput, Double, Double] {
  def zero: Double = 0.0
  
  def reduce(buffer: Double, input: SalesInput): Double = {
    val westernState = (input.state_id >= 10) && (input.state_id <= 19)
    val sales = input.sales
    if ((westernState && (sales > 1000.0)) || ((!westernState) && (sales > 400.0))) {
      buffer + sales
    } else {
      buffer
    }
  }
  
  def finish(reduction: Double): Double = reduction
  
  def bufferEncoder: Encoder[Double] = Encoders.scalaDouble
  def outputEncoder: Encoder[Double] = Encoders.scalaDouble
}