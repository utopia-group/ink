package org.opencypher.morpheus.impl.temporal

import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.sql.expressions.Aggregator
import org.apache.spark.unsafe.types.CalendarInterval
import org.opencypher.morpheus.impl.temporal.TemporalConversions._

case class CalendarInterval(months: Int, microseconds: Long)
case class DurationBuffer(sum: CalendarInterval, count: Long)

object TemporalAggregators {
  class DurationAvg extends Aggregator[CalendarInterval, DurationBuffer, CalendarInterval] {
    def zero: DurationBuffer = DurationBuffer(CalendarInterval(0, 0L), 0L)
    
    def reduce(buffer: DurationBuffer, input: CalendarInterval): DurationBuffer = {
      val newSum = CalendarInterval(
        buffer.sum.months + input.months,
        buffer.sum.microseconds + input.microseconds
      )
      DurationBuffer(newSum, buffer.count + 1)
    }
    
    def merge(buffer1: DurationBuffer, buffer2: DurationBuffer): DurationBuffer = {
      val newSum = CalendarInterval(
        buffer1.sum.months + buffer2.sum.months,
        buffer1.sum.microseconds + buffer2.sum.microseconds
      )
      DurationBuffer(newSum, buffer1.count + buffer2.count)
    }
    
    def finish(reduction: DurationBuffer): CalendarInterval = {
      if (reduction.count == 0) {
        CalendarInterval(0, 0L)
      } else {
        CalendarInterval(
          (reduction.sum.months / reduction.count).toInt,
          reduction.sum.microseconds / reduction.count
        )
      }
    }
    
    def bufferEncoder: Encoder[DurationBuffer] = Encoders.kryo[DurationBuffer]
    def outputEncoder: Encoder[CalendarInterval] = Encoders.kryo[CalendarInterval]
  }
}