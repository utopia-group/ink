package net.jgp.books.spark.ch15.lab400_udaf

import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.sql.expressions.Aggregator

class PointAttributionScalaUdaf extends Aggregator[Int, Int, Int] {
  
  
  def zero: Int = 0
  
  def reduce(buffer: Int, input: Int): Int = {
    val MAX_POINT_PER_ORDER = 3
    val points = if (input < MAX_POINT_PER_ORDER) input else MAX_POINT_PER_ORDER
    buffer + points
  }
  
  def finish(buffer: Int): Int = buffer
  
  def bufferEncoder: Encoder[Int] = Encoders.scalaInt
  def outputEncoder: Encoder[Int] = Encoders.scalaInt
} 