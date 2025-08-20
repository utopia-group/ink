package com.linkedin.feathr.offline.generation.aggregations

import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.sql.expressions.Aggregator

case class EmbeddingInput(value: Seq[Double])
case class MinPoolingBuffer(agg: Seq[Double])

class MinPoolingUDAF(embeddingSize: Int) extends Aggregator[EmbeddingInput, MinPoolingBuffer, Seq[Double]] {
  
  def zero: MinPoolingBuffer = MinPoolingBuffer(Seq.fill(embeddingSize)(Double.MaxValue))
  
  def reduce(buffer: MinPoolingBuffer, input: EmbeddingInput): MinPoolingBuffer = {
    Option(input.value) match {
      case Some(embedding) =>
        if (embedding.size != embeddingSize) {
          throw new RuntimeException(
            s"embedding vector size has a length of ${embedding.size}, different from expected size $embeddingSize")
        }
        val newAgg = buffer.agg.zip(embedding).map { case (x, y) => if (x < y) x else y }
        MinPoolingBuffer(newAgg)
      case None => buffer
    }
  }
  
  def merge(b1: MinPoolingBuffer, b2: MinPoolingBuffer): MinPoolingBuffer = {
    throw new UnsupportedOperationException("merge not implemented")
  }
  
  def finish(reduction: MinPoolingBuffer): Seq[Double] = reduction.agg
  
  def bufferEncoder: Encoder[MinPoolingBuffer] = Encoders.product
  def outputEncoder: Encoder[Seq[Double]] = Encoders.kryo[Seq[Double]]
} 