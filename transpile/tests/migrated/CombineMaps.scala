package sql.udaf.demo1

import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.sql.expressions.Aggregator

case class MapInput[K, V](map: Map[K, V])
case class MapBuffer[K, V](map: Map[K, V])

class CombineMaps[K, V](merge: (V, V) => V) extends Aggregator[MapInput[K, V], MapBuffer[K, V], Map[K, V]] {
  
  def zero: MapBuffer[K, V] = MapBuffer(Map.empty[K, V])
  
  def reduce(buffer: MapBuffer[K, V], input: MapInput[K, V]): MapBuffer[K, V] = {
    val combined = buffer.map ++ input.map.map { 
      case (k, v) => k -> buffer.map.get(k).map(merge(v, _)).getOrElse(v) 
    }
    MapBuffer(combined)
  }
  
  def merge(b1: MapBuffer[K, V], b2: MapBuffer[K, V]): MapBuffer[K, V] = {
    throw new UnsupportedOperationException("merge not implemented")
  }
  
  def finish(reduction: MapBuffer[K, V]): Map[K, V] = reduction.map
  
  def bufferEncoder: Encoder[MapBuffer[K, V]] = Encoders.kryo[MapBuffer[K, V]]
  def outputEncoder: Encoder[Map[K, V]] = Encoders.kryo[Map[K, V]]
} 