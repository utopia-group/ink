package org.aja.tej.examples.sparksql.sql

import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.sql.expressions.Aggregator

class ScalaAggregateFunction extends Aggregator[Double, Double, Double] {
  
  def zero: Double = 0.0
  
  def reduce(buffer: Double, sales: Double): Double = 
    if (sales > 500.0) buffer + sales else buffer
  
  def finish(buffer: Double): Double = buffer
  
  def bufferEncoder: Encoder[Double] = Encoders.scalaDouble
  def outputEncoder: Encoder[Double] = Encoders.scalaDouble
} 