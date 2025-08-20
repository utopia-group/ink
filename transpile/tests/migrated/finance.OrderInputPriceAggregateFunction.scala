package online.javabook.flink.framework.window.function.aggregate

import online.javabook.flink.framework.data.domain.{OrderOutput}
import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.sql.expressions.Aggregator

case class OrderAccumulator(name: String, total: Double, count: Int)
case class OrderInput(name: String, price: Double)


class OrderInputPriceAggregateFunction extends Aggregator[OrderInput, OrderAccumulator, OrderOutput] {
  
  def zero: OrderAccumulator = OrderAccumulator("", 0.0, 0)
  
  def reduce(acc: OrderAccumulator, value: OrderInput): OrderAccumulator = 
    OrderAccumulator(
      name = value.name,
      total = acc.total + value.price,
      count = acc.count + 1
    )
  
  def finish(acc: OrderAccumulator): OrderOutput = {
    val output = new OrderOutput()
    output.setName(acc.name)
    output.setTotal(acc.total)
    output.setCount(acc.count)
    output.setAverage(acc.total / acc.count)
    output
  }
  
  def bufferEncoder: Encoder[OrderAccumulator] = Encoders.product
  def outputEncoder: Encoder[OrderOutput] = Encoders.product
} 