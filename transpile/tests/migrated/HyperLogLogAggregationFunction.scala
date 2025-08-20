package ee.ut.cs.dsg.StreamCardinality.ApproximateCardinalityAggregateFunction

import ee.ut.cs.dsg.StreamCardinality.ApproximateCardinality.HyperLogLog
import ee.ut.cs.dsg.StreamCardinality.ExperimentConfiguration
import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.sql.expressions.Aggregator

case class HyperLogLogAccumulator(f0: Long, f1: String, acc: HyperLogLog) {
  def this() = this(0L, "", new HyperLogLog(8))
}

case class HyperLogLogInput(f0: Long, f1: String, f2: Long)
case class HyperLogLogOutput(f0: Long, f1: String, f2: Long, f3: Long)

class HyperLogLogAggregationFunction extends Aggregator[HyperLogLogInput, HyperLogLogAccumulator, HyperLogLogOutput] {
  
  def zero: HyperLogLogAccumulator = HyperLogLogAccumulator(0L, "", new HyperLogLog(8))
  
  def reduce(acc: HyperLogLogAccumulator, value: HyperLogLogInput): HyperLogLogAccumulator = {
    if (ExperimentConfiguration.experimentType == ExperimentConfiguration.ExperimentType.Latency) {
      val curr = System.nanoTime().toString
      ExperimentConfiguration.async.hset(s"${value.f0}|${value.f1}|$curr", "insertion_start", System.nanoTime().toString)
      val newAcc = acc.copy(f0 = value.f0, f1 = value.f1)
      newAcc.acc.offer(value.f2)
      ExperimentConfiguration.async.hset(s"${value.f0}|${value.f1}|$curr", "insertion_end", System.nanoTime().toString)
      newAcc
    } else {
      val newAcc = acc.copy(f0 = value.f0, f1 = value.f1)
      newAcc.acc.offer(value.f2)
      newAcc
    }
  }
  
  def finish(acc: HyperLogLogAccumulator): HyperLogLogOutput = {
    val cardinality = try {
      acc.acc.cardinality()
    } catch {
      case e: Exception =>
        e.printStackTrace()
        0L
    }
    
    val f3 = ExperimentConfiguration.experimentType match {
      case ExperimentConfiguration.ExperimentType.Latency => System.nanoTime()
      case ExperimentConfiguration.ExperimentType.Throughput => acc.acc.getCount.toLong
      case _ => 0L
    }
    
    HyperLogLogOutput(acc.f0, acc.f1, cardinality, f3)
  }
  
  def bufferEncoder: Encoder[HyperLogLogAccumulator] = Encoders.product
  def outputEncoder: Encoder[HyperLogLogOutput] = Encoders.product
} 