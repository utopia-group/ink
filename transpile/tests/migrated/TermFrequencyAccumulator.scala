package uk.ac.gla.dcs.bigdata.studentstructures

import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.sql.expressions.Aggregator

case class TermFrequencyInput(terms: Map[String, Int])
case class TermFrequencyBuffer(termCounts: Map[String, Int])

class TermFrequencyAccumulator extends Aggregator[TermFrequencyInput, TermFrequencyBuffer, Map[String, Int]] {
  
  def zero: TermFrequencyBuffer = TermFrequencyBuffer(Map.empty[String, Int])
  
  def reduce(buffer: TermFrequencyBuffer, input: TermFrequencyInput): TermFrequencyBuffer = {
    val combined = input.terms.foldLeft(buffer.termCounts) { 
      case (acc, (key, value)) => acc + (key -> (acc.getOrElse(key, 0) + value))
    }
    TermFrequencyBuffer(combined)
  }
  
  def merge(b1: TermFrequencyBuffer, b2: TermFrequencyBuffer): TermFrequencyBuffer = {
    val combined = b2.termCounts.foldLeft(b1.termCounts) {
      case (acc, (key, value)) => acc + (key -> (acc.getOrElse(key, 0) + value))
    }
    TermFrequencyBuffer(combined)
  }
  
  def finish(reduction: TermFrequencyBuffer): Map[String, Int] = reduction.termCounts
  
  def bufferEncoder: Encoder[TermFrequencyBuffer] = Encoders.product
  def outputEncoder: Encoder[Map[String, Int]] = Encoders.kryo[Map[String, Int]]
} 