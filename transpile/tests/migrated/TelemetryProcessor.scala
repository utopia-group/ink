package jbcodeforce.domain

import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.sql.expressions.Aggregator

case class AccessLogEvent(dt: String, username: String, executeSql: String, path: String, location: String, dataRows: Long, cosTime: Long)
case class AccessMiddleResult(dt: String, username: String, executeSql: String, path: String, location: String, maxDataRows: Long, maxCosTime: Long)

class AggAccessMiddleResult extends Aggregator[AccessLogEvent, AccessMiddleResult, AccessMiddleResult] {
  
  def zero: AccessMiddleResult = AccessMiddleResult("", "", "", "", "", 0L, 0L)
  
  def reduce(accumulator: AccessMiddleResult, value: AccessLogEvent): AccessMiddleResult = {
    AccessMiddleResult(
      value.dt,
      value.username,
      value.executeSql,
      value.path,
      value.location,
      max(value.dataRows, accumulator.maxDataRows),
      max(value.cosTime, accumulator.maxCosTime)
    )
  }
  
  def merge(a: AccessMiddleResult, b: AccessMiddleResult): AccessMiddleResult = {
    a
  }
  
  def finish(accumulator: AccessMiddleResult): AccessMiddleResult = {
    AccessMiddleResult(
      accumulator.dt,
      accumulator.username,
      accumulator.executeSql,
      accumulator.path,
      accumulator.location,
      accumulator.maxDataRows,
      accumulator.maxCosTime
    )
  }
  
  def bufferEncoder: Encoder[AccessMiddleResult] = Encoders.product
  def outputEncoder: Encoder[AccessMiddleResult] = Encoders.product
} 
