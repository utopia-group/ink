package com.amazonaws.kaja.samples

import samples.clickstream.avro.ClickEvent
import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.sql.expressions.Aggregator
import scala.collection.mutable

case class ClickEventAccumulator(
  userId: Int,
  eventCount: Int,
  eventCountWithOrderCheckout: Int,
  departmentsVisited: Set[String]
)

case class ClickStreamResult(
  userId: Int,
  eventCount: Int,
  eventCountWithOrderCheckout: Int,
  departmentsVisited: Set[String],
  eventKey: Int
)

case class ClickEvent(
  userid: Int,
  productType: String,
  eventType: String
)

class ClickstreamAggregate extends Aggregator[ClickEvent, ClickEventAccumulator, ClickStreamResult] {
  
  def zero: ClickEventAccumulator = ClickEventAccumulator(0, 0, 0, Set.empty[String])
  
  def reduce(acc: ClickEventAccumulator, value: ClickEvent): ClickEventAccumulator = {
    val newEventCount = 
      if (value.productType != "N/A") 
        acc.eventCount + 1
      else 
        acc.eventCount

    val newDepartments = 
      if (value.productType != "N/A") 
        acc.departmentsVisited + value.productType
      else 
        acc.departmentsVisited
    
    val newUserId = if (acc.userId == 0) value.userid else acc.userId
    
    val newEventCountWithOrderCheckout = 
      if (value.eventType == "order_checkout") newEventCount else acc.eventCountWithOrderCheckout
    
    ClickEventAccumulator(newUserId, newEventCount, newEventCountWithOrderCheckout, newDepartments)
  }
  
  def finish(acc: ClickEventAccumulator): ClickStreamResult = 
    ClickStreamResult(acc.userId, acc.eventCount, acc.eventCountWithOrderCheckout, acc.departmentsVisited, 1)
  
  def bufferEncoder: Encoder[ClickEventAccumulator] = Encoders.product
  def outputEncoder: Encoder[ClickStreamResult] = Encoders.product
}