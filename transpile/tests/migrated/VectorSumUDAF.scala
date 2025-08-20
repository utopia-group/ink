package de.tudarmstadt.lt.wsd.pipeline.sql

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.sql.expressions.Aggregator

case class VectorInput(v: Vector)
case class VectorBuffer(map: Map[Int, Double], size: Int)

class VectorSumUDAF extends Aggregator[VectorInput, VectorBuffer, Vector] {
  
  def zero: VectorBuffer = VectorBuffer(Map.empty[Int, Double], -1)
  
  def reduce(buffer: VectorBuffer, input: VectorInput): VectorBuffer = {
    val vector = input.v.toSparse
    val vectorMap = vector.indices.zip(vector.values).toMap
    val mergedMap = VectorSumUDAF.mergeMaps(buffer.map, vectorMap)
    VectorBuffer(mergedMap, vector.size)
  }
  
  def merge(b1: VectorBuffer, b2: VectorBuffer): VectorBuffer = {
    throw new UnsupportedOperationException("merge not implemented")
  }
  
  def finish(reduction: VectorBuffer): Vector = {
    Vectors.sparse(reduction.size, reduction.map.toSeq)
  }
  
  def bufferEncoder: Encoder[VectorBuffer] = Encoders.product
  def outputEncoder: Encoder[Vector] = Encoders.kryo[Vector]
}

object VectorSumUDAF {
  def mergeMaps(map1: Map[Int, Double], map2: Map[Int, Double]): Map[Int, Double] = {
    map1.foldLeft(map2) {
      case (to, (k, v)) => to + (k -> to.get(k).map(_ + v).getOrElse(v))
    }
  }
} 