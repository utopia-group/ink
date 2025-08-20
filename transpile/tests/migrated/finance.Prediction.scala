package com.nyu.bigData

import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.sql.expressions.Aggregator

case class OBVInput(volume: Long, closeYc: Double)

class OBV extends Aggregator[OBVInput, Long, Long] {
  
  def zero: Long = 0L
  
  def reduce(buffer: Long, input: OBVInput): Long = 
    if (input.closeYc > 0) buffer + input.volume else buffer - input.volume
  
  def finish(buffer: Long): Long = buffer
  
  def bufferEncoder: Encoder[Long] = Encoders.scalaLong
  def outputEncoder: Encoder[Long] = Encoders.scalaLong
} 