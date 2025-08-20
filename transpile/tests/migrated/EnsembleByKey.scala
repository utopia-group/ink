package com.microsoft.ml.spark

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.sql.expressions.Aggregator

case class VectorInput(v: Vector)
case class VectorAvgBuffer(buff: Array[Double], count: Long)

class VectorAvg(n: Int) extends Aggregator[VectorInput, VectorAvgBuffer, Vector] {
  
  def zero: VectorAvgBuffer = VectorAvgBuffer(Array.fill(n)(0.0), 0L)
  
  def reduce(buffer: VectorAvgBuffer, input: VectorInput): VectorAvgBuffer = {
    val v = input.v.toSparse
    val newBuff = buffer.buff.clone()
    v.indices.foreach { i =>
      newBuff(i) += v(i)
    }
    VectorAvgBuffer(newBuff, buffer.count + 1L)
  }
  
  def merge(b1: VectorAvgBuffer, b2: VectorAvgBuffer): VectorAvgBuffer = {
    throw new UnsupportedOperationException("merge not implemented")
  }
  
  def finish(reduction: VectorAvgBuffer): Vector = {
    val avgArray = reduction.buff.map(_ / reduction.count)
    Vectors.dense(avgArray)
  }
  
  def bufferEncoder: Encoder[VectorAvgBuffer] = Encoders.kryo[VectorAvgBuffer]
  def outputEncoder: Encoder[Vector] = Encoders.kryo[Vector]
} 