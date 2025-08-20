package com.shangbaishuyao.udf

import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.sql.expressions.Aggregator

case class CityNameInput(cityName: String)
case class CityCountBuffer(cityCount: Map[String, Long])
case class CityRatio(cityName: String, cityRatio: Double) {
  override def toString: String = s"$cityName${Math.round(cityRatio * 10.0) / 10.0}%"
}

object CityRatioUDAF extends Aggregator[CityNameInput, CityCountBuffer, String] {
  
  def zero: CityCountBuffer = CityCountBuffer(Map.empty[String, Long])
  
  def reduce(buffer: CityCountBuffer, input: CityNameInput): CityCountBuffer = {
    val newCount = buffer.cityCount.getOrElse(input.cityName, 0L) + 1L
    CityCountBuffer(buffer.cityCount + (input.cityName -> newCount))
  }
  
  def merge(b1: CityCountBuffer, b2: CityCountBuffer): CityCountBuffer = {
    throw new UnsupportedOperationException("merge not implemented")
  }
  
  def finish(reduction: CityCountBuffer): String = {
    val totalCount = reduction.cityCount.values.sum
    val cityToCountTop2 = reduction.cityCount.toList.sortWith(_._2 > _._2).take(2)
    
    var otherRatio = 100.0
    val result = cityToCountTop2.map {
      case (city, count) => 
        val cityRatio = Math.round(count * 10000.0 / totalCount) / 100.0
        otherRatio -= cityRatio
        CityRatio(city, cityRatio)
    }
    
    val ratios = result :+ CityRatio("其他", Math.round(otherRatio * 10.0) / 10.0)
    ratios.mkString(",")
  }
  
  def bufferEncoder: Encoder[CityCountBuffer] = Encoders.product
  def outputEncoder: Encoder[String] = Encoders.STRING
} 