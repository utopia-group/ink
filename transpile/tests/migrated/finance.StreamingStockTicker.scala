package advanced.windowing

import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.sql.expressions.Aggregator

case class StockPrices(ticker: String, price: Double, timestamp: Long)
case class StockAccumulator(ticker: String, sum: Double, count: Long)
case class StockResult(ticker: String, average: Double)

class AverageAggregate extends Aggregator[StockPrices, StockAccumulator, StockResult] {
  
  def zero: StockAccumulator = StockAccumulator("", 0.0, 0L)
  
  def reduce(acc: StockAccumulator, value: StockPrices): StockAccumulator = 
    StockAccumulator(
      ticker = value.ticker,
      sum = acc.sum + value.price,
      count = acc.count + 1L
    )
  
  def finish(acc: StockAccumulator): StockResult = 
    StockResult(acc.ticker, acc.sum / acc.count)
  
  def bufferEncoder: Encoder[StockAccumulator] = Encoders.product
  def outputEncoder: Encoder[StockResult] = Encoders.product
} 