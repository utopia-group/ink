package io.palyvos.provenance.usecases.linearroad.noprovenance

import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.sql.expressions.Aggregator

case class VehicleTuple(
  timestamp: Long,
  vid: Long,
  reports: Int,
  latestXWay: Int,
  latestLane: Int,
  latestDir: Int,
  latestSeg: Int,
  latestPos: Int,
  uniquePosition: Boolean,
  stimulus: Long,
  key: String
) {
  def getTimestamp: Long = timestamp
  def getStimulus: Long = stimulus
  def getVid: Long = vid
}

case class AccidentTuple(
  vids: Set[Long],
  timestamp: Long,
  stimulus: Long,
  xWay: Int,
  lane: Int,
  dir: Int,
  seg: Int,
  pos: Int
)

class LinearRoadAccidentAggregate extends Aggregator[VehicleTuple, AccidentTuple, AccidentTuple] {
  
  def zero: AccidentTuple = AccidentTuple(Set.empty[Long], Long.MinValue, Long.MinValue, 0, 0, 0, 0, 0)
  
  def reduce(acc: AccidentTuple, tuple: VehicleTuple): AccidentTuple = {
    val newVids = acc.vids + tuple.vid
    AccidentTuple(
      vids = newVids,
      timestamp = if (acc.timestamp > tuple.timestamp) acc.timestamp else tuple.timestamp,
      stimulus = if (acc.stimulus > tuple.stimulus) acc.stimulus else tuple.stimulus,
      xWay = tuple.latestXWay,
      lane = tuple.latestLane,
      dir = tuple.latestDir,
      seg = tuple.latestSeg,
      pos = tuple.latestPos
    )
  }
  
  def finish(acc: AccidentTuple): AccidentTuple = acc
  
  def bufferEncoder: Encoder[AccidentTuple] = Encoders.product
  def outputEncoder: Encoder[AccidentTuple] = Encoders.product
} 