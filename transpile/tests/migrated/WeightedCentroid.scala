package vectorpipe.vectortile

import geotrellis.vector._
import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.sql.expressions.Aggregator
import org.locationtech.jts.geom.{Coordinate, GeometryFactory}

case class WeightedPointInput(x: Double, y: Double, weight: Double)
case class WeightedCentroidBuffer(x: Double, y: Double, weight: Double)

class WeightedCentroid extends Aggregator[WeightedPointInput, WeightedCentroidBuffer, Point] {

  def zero: WeightedCentroidBuffer = WeightedCentroidBuffer(0.0, 0.0, 0.0)

  def reduce(buffer: WeightedCentroidBuffer, input: WeightedPointInput): WeightedCentroidBuffer = {
    WeightedCentroidBuffer(
      buffer.x + (input.x * input.weight),
      buffer.y + (input.y * input.weight),
      buffer.weight + input.weight
    )
  }

  def merge(b1: WeightedCentroidBuffer, b2: WeightedCentroidBuffer): WeightedCentroidBuffer = {
    throw new UnsupportedOperationException("merge not implemented")
  }

  def finish(reduction: WeightedCentroidBuffer): Point = {
    val wx = reduction.x
    val wy = reduction.y
    val wt = reduction.weight
    Point((new GeometryFactory).createPoint(new Coordinate(wx / wt, wy / wt)))
  }

  def bufferEncoder: Encoder[WeightedCentroidBuffer] = Encoders.product
  def outputEncoder: Encoder[Point] = Encoders.kryo[Point]
}
