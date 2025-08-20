package com.mozilla.telemetry.utils.udfs

import org.apache.spark.sql.{Encoder, Encoders, Row}
import org.apache.spark.sql.expressions.Aggregator
import org.apache.spark.sql.types._

class AggRowFirst[IndexDataType](schema: StructType, idIndex: Int, indexSqlDataType: DataType)
  extends Aggregator[Seq[Row], Map[IndexDataType, Row], Seq[Row]] {
  
  def zero: Map[IndexDataType, Row] = Map.empty[IndexDataType, Row]
  
  def reduce(buffer: Map[IndexDataType, Row], input: Seq[Row]): Map[IndexDataType, Row] = {
    input.map(row => row.getAs[IndexDataType](idIndex) -> row).toMap ++ buffer
  }
  
  def finish(reduction: Map[IndexDataType, Row]): Seq[Row] = reduction.values.toSeq
  
  def bufferEncoder: Encoder[Map[IndexDataType, Row]] = Encoders.kryo[Map[IndexDataType, Row]]
  def outputEncoder: Encoder[Seq[Row]] = Encoders.kryo[Seq[Row]]
} 