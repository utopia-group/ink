package DataFrames

import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.sql.expressions.Aggregator

class LongestWord extends Aggregator[String, String, String] {
  
  def zero: String = ""
  
  def reduce(buffer: String, input: String): String = 
    if (input.length > buffer.length) input else buffer
  
  def finish(buffer: String): String = buffer
  
  def bufferEncoder: Encoder[String] = Encoders.STRING
  def outputEncoder: Encoder[String] = Encoders.STRING
} 