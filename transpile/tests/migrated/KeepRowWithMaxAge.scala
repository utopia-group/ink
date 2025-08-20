package com.hamshif.wielder.pipelines.snippets

import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.sql.expressions.Aggregator

case class PersonRow(id: Long, name: String, requisite: String, money: Double, age: Int)

class KeepRowWithMaxAge extends Aggregator[PersonRow, PersonRow, PersonRow] {
  
  def zero: PersonRow = PersonRow(0L, "", "", 0.0, Int.MinValue)
  
  def reduce(buffer: PersonRow, input: PersonRow): PersonRow = 
    if (input.age > buffer.age) input else buffer
  
  def finish(buffer: PersonRow): PersonRow = buffer
  
  def bufferEncoder: Encoder[PersonRow] = Encoders.product
  def outputEncoder: Encoder[PersonRow] = Encoders.product
} 