package org.example.queries

import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.sql.expressions.Aggregator

case class Output(mean: Float, min: Long, max: Long, stddev: Float)

case class BidAggregator(sum: Long, count: Long, min: Long, max: Long, sumSquares: Long)
case class Bid(price: Long)


class QWAggregator extends Aggregator[Bid, BidAggregator, Output] {
  
  def zero: BidAggregator = BidAggregator(0L, 0L, Long.MaxValue, Long.MinValue, 0L)
  
  def reduce(acc: BidAggregator, bid: Bid): BidAggregator = 
    BidAggregator(
      sum = acc.sum + bid.price,
      count = acc.count + 1,
      min = if (bid.price < acc.min) bid.price else acc.min,
      max = if (bid.price > acc.max) bid.price else acc.max,
      sumSquares = acc.sumSquares + (bid.price * bid.price)
    )
  
  def finish(acc: BidAggregator): Output = {
    val mean = acc.sum.toFloat / acc.count
    val variance = acc.sumSquares.toFloat / acc.count - mean * mean
    Output(mean, acc.min, acc.max, math.sqrt(variance).toFloat)
  }
  
  def bufferEncoder: Encoder[BidAggregator] = Encoders.product
  def outputEncoder: Encoder[Output] = Encoders.product
} 