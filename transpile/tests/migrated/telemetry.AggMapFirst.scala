package com.mozilla.telemetry.utils.udfs

import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.sql.expressions.Aggregator

class AggMapFirst extends Aggregator[Map[String, Int], Map[String, Int], Map[String, Int]] {
  def zero: Map[String, Int] = Map.empty[String, String]
  
  def reduce(buffer: Map[String, Int], input: Map[String, Int]): Map[String, Int] = {
    input.filter(_._2 != 0) ++ buffer
  }
  
  def finish(reduction: Map[String, Int]): Map[String, Int] = reduction
  
  def bufferEncoder: Encoder[Map[String, Int]] = Encoders.kryo[Map[String, Int]]
  def outputEncoder: Encoder[Map[String, Int]] = Encoders.kryo[Map[String, Int]]
} 