package com.sketches.spark.theta.udaf

import com.yahoo.memory.{Memory, WritableMemory}
import com.yahoo.sketches.theta.{SetOperation, Union}
import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.sql.expressions.Aggregator

case class SketchInput(sketch: Array[Byte])
case class UnionSketchBuffer(unionSketch: Option[Array[Byte]])

class UnionSketchUDAF extends Aggregator[SketchInput, UnionSketchBuffer, Array[Byte]] {
  
  def zero: UnionSketchBuffer = UnionSketchBuffer(None)
  
  def reduce(buffer: UnionSketchBuffer, input: SketchInput): UnionSketchBuffer = {
    val inputMemorySketch = Memory.wrap(input.sketch)
    
    buffer.unionSketch match {
      case None =>
        val unionSketch = SetOperation.builder.buildUnion
        unionSketch.update(inputMemorySketch)
        UnionSketchBuffer(Some(unionSketch.toByteArray))
      case Some(existingSketch) =>
        val unionMemorySketch = WritableMemory.wrap(existingSketch)
        val unionSketch = SetOperation.wrap(unionMemorySketch).asInstanceOf[Union]
        unionSketch.update(inputMemorySketch)
        UnionSketchBuffer(Some(unionSketch.toByteArray))
    }
  }
  
  def merge(b1: UnionSketchBuffer, b2: UnionSketchBuffer): UnionSketchBuffer = {
    throw new UnsupportedOperationException("merge not implemented")
  }
  
  def finish(reduction: UnionSketchBuffer): Array[Byte] = {
    reduction.unionSketch match {
      case Some(sketchBytes) =>
        val memorySketch = Memory.wrap(sketchBytes)
        val sketch = SetOperation.wrap(memorySketch).asInstanceOf[Union].getResult
        sketch.toByteArray
      case None => Array.empty[Byte]
    }
  }
  
  def bufferEncoder: Encoder[UnionSketchBuffer] = Encoders.kryo[UnionSketchBuffer]
  def outputEncoder: Encoder[Array[Byte]] = Encoders.kryo[Array[Byte]]
} 