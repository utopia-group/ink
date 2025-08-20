package ee.ut.cs.dsg.StreamCardinality.ApproximateCardinalityAggregateFunction

import ee.ut.cs.dsg.StreamCardinality.ExperimentConfiguration
import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.sql.expressions.Aggregator

case class BloomFilterAccumulator(f0: Long, f1: String, acc: Set[Long]) {
  def this() = this(0L, "", Set.empty[Long])
}

case class InputTuple(f0: Long, f1: String, f2: Long)
case class OutputTuple(f0: Long, f1: String, f2: Long, f3: Long)

class BloomFilterAggregationFunction extends Aggregator[InputTuple, BloomFilterAccumulator, OutputTuple] {
  
  def zero: BloomFilterAccumulator = BloomFilterAccumulator(0L, "", Set.empty[Long])
  
  def reduce(acc: BloomFilterAccumulator, value: InputTuple): BloomFilterAccumulator = {
    BloomFilterAccumulator(value.f0, value.f1, acc.acc + value.f2)
  }
  
  def finish(acc: BloomFilterAccumulator): OutputTuple = {
    val f3 = ExperimentConfiguration.experimentType match {
      case ExperimentConfiguration.ExperimentType.Latency => System.nanoTime()
      case ExperimentConfiguration.ExperimentType.Throughput => acc.acc.size.toLong
      case _ => 0L
    }
    OutputTuple(acc.f0, acc.f1, acc.acc.size.toLong, f3)
  }
  
  def bufferEncoder: Encoder[BloomFilterAccumulator] = Encoders.product
  def outputEncoder: Encoder[OutputTuple] = Encoders.product
}
