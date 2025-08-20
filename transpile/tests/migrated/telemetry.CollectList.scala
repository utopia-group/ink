package com.mozilla.telemetry.utils.udfs

import org.apache.spark.sql.{Encoder, Encoders, Row}
import org.apache.spark.sql.expressions.Aggregator
import org.apache.spark.sql.types._

class CollectList(inputStruct: StructType, orderCols: List[String], maxLength: Option[Int]) 
  extends Aggregator[Row, Seq[Seq[Any]], Row] {
  
  private val numFields = inputStruct.fields.length
  
  def zero: Seq[Seq[Any]] = (0 until numFields).map(_ => Seq.empty[Any])
  
  def reduce(buffer: Seq[Seq[Any]], input: Row): Seq[Seq[Any]] = {
    (0 until numFields).map(n => buffer(n) :+ input(n))
  }
  
  def finish(reduction: Seq[Seq[Any]]): Row = {
    val sortOrder = (0 until reduction.head.length).toList
    val sortedArrays = (0 until numFields).map { n =>
      val sorted = sortOrder.map(reduction(n))
      maxLength.map(sorted.take).getOrElse(sorted)
    }
    Row.fromSeq(sortedArrays)
  }
  
  def bufferEncoder: Encoder[Seq[Seq[Any]]] = Encoders.kryo[Seq[Seq[Any]]]
  def outputEncoder: Encoder[Row] = Encoders.kryo[Row]
} 