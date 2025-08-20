package org.locationtech.rasterframes.expressions.aggregates

import org.locationtech.rasterframes._
import org.locationtech.rasterframes.stats.LocalCellStatistics
import geotrellis.raster.{Tile, DoubleConstantNoDataCellType, IntConstantNoDataCellType, IntUserDefinedNoDataCellType}
import geotrellis.raster.mapalgebra.local._
import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.sql.expressions.Aggregator

case class LocalStatsBuffer(
  count: Option[Tile],
  min: Option[Tile],
  max: Option[Tile],
  sum: Option[Tile],
  sumSqr: Option[Tile]
)

case class LocalStatsResult(
  count: Tile,
  min: Tile,
  max: Tile,
  mean: Tile,
  variance: Tile
)

class LocalStatsAggregate extends Aggregator[Tile, LocalStatsBuffer, LocalStatsResult] {
  
  def zero: LocalStatsBuffer = LocalStatsBuffer(None, None, None, None, None)
  
  def reduce(buffer: LocalStatsBuffer, tile: Tile): LocalStatsBuffer = {
    if (tile != null) {
      LocalStatsBuffer(
        count = buffer.count.map(c => Add(c, Defined(tile))).orElse(Some(Defined(tile).convert(IntConstantNoDataCellType))),
        min = buffer.min.map(m => Min(m, tile)).orElse(Some(tile)),
        max = buffer.max.map(m => Max(m, tile)).orElse(Some(tile)),
        sum = buffer.sum.map(s => Add(s, tile)).orElse(Some(tile.convert(DoubleConstantNoDataCellType))),
        sumSqr = buffer.sumSqr.map(s => {
          val d = tile.convert(DoubleConstantNoDataCellType)
          Add(s, Multiply(d, d))
        }).orElse({
          val d = tile.convert(DoubleConstantNoDataCellType)
          Some(Multiply(d, d))
        })
      )
    } else {
      buffer
    }
  }
  
  def finish(buffer: LocalStatsBuffer): LocalStatsResult = {
    buffer.count match {
      case Some(cnt) =>
        val count = cnt.interpretAs(IntUserDefinedNoDataCellType(0))
        val sum = buffer.sum.get
        val sumSqr = buffer.sumSqr.get
        val mean = Divide(sum, count)
        val meanSqr = Multiply(mean, mean)
        val variance = Subtract(Divide(sumSqr, count), meanSqr)
        LocalStatsResult(count, buffer.min.get, buffer.max.get, mean, variance)
      case None =>
        null.asInstanceOf[LocalStatsResult]
    }
  }
  
  def bufferEncoder: Encoder[LocalStatsBuffer] = Encoders.product
  def outputEncoder: Encoder[LocalStatsResult] = Encoders.product
} 