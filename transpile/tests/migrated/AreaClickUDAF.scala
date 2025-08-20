package com.atguigu.acc

import java.text.DecimalFormat
import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.sql.expressions.Aggregator

case class CityInput(city: String)
case class CityBuffer(cityCountMap: Map[String, Long], totalCount: Long)

class AreaClickUDAF extends Aggregator[CityInput, CityBuffer, String] {
  
  def zero: CityBuffer = CityBuffer(Map.empty[String, Long], 0L)
  
  def reduce(buffer: CityBuffer, input: CityInput): CityBuffer = {
    val updatedMap = buffer.cityCountMap + (input.city -> (buffer.cityCountMap.getOrElse(input.city, 0L) + 1L))
    CityBuffer(updatedMap, buffer.totalCount + 1L)
  }
  
  def merge(b1: CityBuffer, b2: CityBuffer): CityBuffer = {
    throw new UnsupportedOperationException("merge not implemented")
  }
  
  def finish(reduction: CityBuffer): String = {
    val cityRate = reduction.cityCountMap.toList.sortBy(-_._2).take(2).map {
      case (cityName, count) => CityRemark(cityName, count.toDouble / reduction.totalCount)
    }
    val otherRate = cityRate.foldLeft(1.0)(_ - _.rate)
    val allRates = cityRate :+ CityRemark("其他", otherRate)
    allRates.mkString(", ")
  }
  
  def bufferEncoder: Encoder[CityBuffer] = Encoders.product
  def outputEncoder: Encoder[String] = Encoders.STRING
}

case class CityRemark(cityName: String, rate: Double) {
  val f = new DecimalFormat("0.00%")
  override def toString: String = s"$cityName:${f.format(rate)}"
} 