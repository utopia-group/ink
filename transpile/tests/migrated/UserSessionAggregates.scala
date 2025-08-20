package tutorial.buildon.aws.streaming.flink

import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.sql.expressions.Aggregator

case class UserIdSessionEvent(orderCheckoutEventCount: Int)
case class SessionBuffer(totalEvents: Int, checkoutEvents: Int)

class UserSessionAggregates extends Aggregator[UserIdSessionEvent, SessionBuffer, (Int, Int, Int)] {
  
  def zero: SessionBuffer = SessionBuffer(0, 0)
  
  def reduce(buffer: SessionBuffer, value: UserIdSessionEvent): SessionBuffer = {
    val newCheckoutEvents = if (value.orderCheckoutEventCount != 0) buffer.checkoutEvents + 1 else buffer.checkoutEvents
    SessionBuffer(buffer.totalEvents + 1, newCheckoutEvents)
  }
  
  def merge(b1: SessionBuffer, b2: SessionBuffer): SessionBuffer = {
    throw new UnsupportedOperationException("merge not implemented")
  }
  
  def finish(reduction: SessionBuffer): (Int, Int, Int) = {
    val percentage = if (reduction.totalEvents > 0) reduction.checkoutEvents * 100 / reduction.totalEvents else 0
    (reduction.totalEvents, reduction.checkoutEvents, percentage)
  }
  
  def bufferEncoder: Encoder[SessionBuffer] = Encoders.product
  def outputEncoder: Encoder[(Int, Int, Int)] = Encoders.tuple(Encoders.INT, Encoders.INT, Encoders.INT)
} 