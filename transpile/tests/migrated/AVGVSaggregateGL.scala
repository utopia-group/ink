package streamingRetention.usecases.linearRoad

import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.sql.expressions.Aggregator

case class LinearRoadInputTuple(seg: Int, dir: Int, vid: Long, timestamp: Long, stimulus: Long, speed: Double)

case class SpeedAccumulator(timestamp: Long, stimulus: Long, count: Int, speed: Double, seg: Int, dir: Int, vid: Long)

class AVGVSaggregateGL extends Aggregator[LinearRoadInputTuple, SpeedAccumulator, LavTuple] {
  
  def zero: SpeedAccumulator = SpeedAccumulator(0L, 0L, 0, 0.0, 0, 0, 0L)
  
  def reduce(acc: SpeedAccumulator, tuple: LinearRoadInputTuple): SpeedAccumulator = {
    val newTimestamp = if (acc.timestamp > tuple.timestamp) acc.timestamp else tuple.timestamp
    val newStimulus = if (acc.stimulus > tuple.stimulus) acc.stimulus else tuple.stimulus
    val newSpeed = acc.speed + tuple.speed
    val newCount = acc.count + 1
    val newSeg = tuple.seg
    val newDir = tuple.dir
    val newVid = tuple.vid
    
    SpeedAccumulator(newTimestamp, newStimulus, newCount, newSpeed, newSeg, newDir, newVid)
  }
  
  def finish(acc: SpeedAccumulator): LavTuple = 
    LavTuple(acc.timestamp, s"${acc.seg}:${acc.dir}:${acc.vid}", acc.stimulus, 
              if (acc.count > 0) acc.speed / acc.count else 0.0)
  
  def bufferEncoder: Encoder[SpeedAccumulator] = Encoders.product
  def outputEncoder: Encoder[LavTuple] = Encoders.product
}

