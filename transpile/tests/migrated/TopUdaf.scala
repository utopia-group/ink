package tech.ytsaurus.spyt.common.utils

import org.apache.spark.sql.{Encoder, Encoders, Row}
import org.apache.spark.sql.expressions.Aggregator
import org.apache.spark.sql.types.StructType

case class TopInput(row: Row)
case class TopBuffer(currentRow: Option[Row])

class TopUdaf(schema: StructType, topColumns: Seq[String]) extends Aggregator[TopInput, TopBuffer, Row] {
  private val topIndices = topColumns.map(schema.fieldIndex)
  
  def zero: TopBuffer = TopBuffer(None)
  
  def reduce(buffer: TopBuffer, input: TopInput): TopBuffer = {
    import Ordering.Implicits._
    val candidate = topIndices.map(input.row.getAs[String])
    
    buffer.currentRow match {
      case None => TopBuffer(Some(input.row))
      case Some(current) =>
        val currentValues = topIndices.map(current.getAs[String])
        if (candidate < currentValues) {
          TopBuffer(Some(input.row))
        } else {
          buffer
        }
    }
  }
  
  def merge(b1: TopBuffer, b2: TopBuffer): TopBuffer = {
    throw new UnsupportedOperationException("merge not implemented")
  }
  
  def finish(reduction: TopBuffer): Row = {
    reduction.currentRow.orNull
  }
  
  def bufferEncoder: Encoder[TopBuffer] = Encoders.kryo[TopBuffer]
  def outputEncoder: Encoder[Row] = Encoders.kryo[Row]
} 