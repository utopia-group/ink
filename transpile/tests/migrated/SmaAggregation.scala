package org.streambench

import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.sql.expressions.Aggregator

case class Data(start_time: Long, end_time: Long, payload: Long)
case class SmaBuffer(data: Data, count: Long)

class SmaAggregation extends Aggregator[Data, SmaBuffer, Data] {
  
  def zero: SmaBuffer = SmaBuffer(Data(Integer.MAX_VALUE.toLong,  Integer.MIN_VALUE.toLong, 0L), 0L)
  
  def reduce(buffer: SmaBuffer, value: Data): SmaBuffer = {
    val newStartTime = if (value.start_time < buffer.data.start_time) value.start_time else buffer.data.start_time
    val newEndTime = if (value.end_time > buffer.data.end_time) value.end_time else buffer.data.end_time
    val newPayload = buffer.data.payload + value.payload
    val newCount = buffer.count + 1
    SmaBuffer(Data(newStartTime, newEndTime, newPayload), newCount)
  }
  
  def merge(b1: SmaBuffer, b2: SmaBuffer): SmaBuffer = {
    throw new UnsupportedOperationException("merge not implemented")
  }
  
  def finish(reduction: SmaBuffer): Data = {
    Data(reduction.data.start_time, reduction.data.end_time, reduction.data.payload / reduction.count)
  }
  
  def bufferEncoder: Encoder[SmaBuffer] = Encoders.product
  def outputEncoder: Encoder[Data] = Encoders.product
} 