package org.heigit.osmalert.flinkjobjar

import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.sql.expressions.Aggregator

case class Changeset(userId: Int)
case class Contribution(changeset: Changeset)
case class StatsAccumulator(count: Int, uniqueUsers: Set[Int])
case class StatsResult(count: Int, uniqueUserCount: Int)

class StatsAggregateFunction extends Aggregator[Contribution, StatsAccumulator, StatsResult] {
  
  def zero: StatsAccumulator = StatsAccumulator(0, Set.empty[Int])
  
  def reduce(buffer: StatsAccumulator, value: Contribution): StatsAccumulator = {
    val updatedUsers = buffer.uniqueUsers + value.changeset.userId
    StatsAccumulator(buffer.count + 1, updatedUsers)
  }
  
  def merge(b1: StatsAccumulator, b2: StatsAccumulator): StatsAccumulator = {
    throw new UnsupportedOperationException("merge not implemented")
  }
  
  def finish(reduction: StatsAccumulator): StatsResult = {
    StatsResult(reduction.count, reduction.uniqueUsers.size)
  }
  
  def bufferEncoder: Encoder[StatsAccumulator] = Encoders.product
  def outputEncoder: Encoder[StatsResult] = Encoders.product
} 