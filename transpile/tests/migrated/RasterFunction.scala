package dbis.stark.sql.raster

import dbis.stark.raster.{Bucket, RasterUtils, Tile}
import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.sql.expressions.Aggregator

class CalcRasterHistogram extends Aggregator[Tile[Double], Option[Seq[Bucket[Double]]], Seq[Bucket[Double]]] {
  
  private val bucketCount = 10
  
  def zero: Option[Seq[Bucket[Double]]] = None
  
  def reduce(buffer: Option[Seq[Bucket[Double]]], tile: Tile[Double]): Option[Seq[Bucket[Double]]] = {
    val histo = tile.histogram(bucketCount)
    buffer match {
      case None => Some(histo)
      case Some(existing) => Some(RasterUtils.combineHistograms[Double](existing, histo))
    }
  }
  
  def finish(buffer: Option[Seq[Bucket[Double]]]): Seq[Bucket[Double]] = 
    buffer.getOrElse(Seq.empty)
  
  def bufferEncoder: Encoder[Option[Seq[Bucket[Double]]]] = Encoders.product
  def outputEncoder: Encoder[Seq[Bucket[Double]]] = Encoders.product
} 