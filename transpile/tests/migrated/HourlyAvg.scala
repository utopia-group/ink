package com.spark.scala.sbt.sparkudf

import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.sql.expressions.Aggregator

case class ParsedTimestamp(year: Int, month: Int, day: Int, dayOfWeek: Int, weekOfYear: Int, hour: Int, minute: Int, second: Int, nano: Int)

case class HourlyBuffer(
  totalBookings: Int,
  distinctMonthCount: Int,
  previousYear: Int,
  currentYear: Int,
  previousMonth: Int,
  currentMonth: Int,
  previousWeek: Int,
  currentWeek: Int,
  previousDay: Int,
  currentDay: Int,
  previousHour: Int,
  currentHour: Int
)

class HourlyAvg extends Aggregator[ParsedTimestamp, HourlyBuffer, Double] {

  def zero: HourlyBuffer = HourlyBuffer(0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

  def reduce(buffer: HourlyBuffer, input: ParsedTimestamp): HourlyBuffer = {
    val isFirstUpdate = buffer.currentWeek == 0
    
    val newTotalBookings = buffer.totalBookings + 1
    val newPreviousYear = if (isFirstUpdate) buffer.previousYear else buffer.currentYear
    val newCurrentYear = input.year
    val newPreviousMonth = if (isFirstUpdate) buffer.previousMonth else buffer.currentMonth
    val newCurrentMonth = input.month
    val newPreviousWeek = if (isFirstUpdate) buffer.previousWeek else buffer.currentWeek
    val newCurrentWeek = input.weekOfYear
    val newPreviousDay = if (isFirstUpdate) buffer.previousDay else buffer.currentDay
    val newCurrentDay = input.dayOfWeek
    val newPreviousHour = if (isFirstUpdate) buffer.previousHour else buffer.currentHour
    val newCurrentHour = input.hour

    val hasTimeChanged = (newCurrentHour != newPreviousHour) ||
                        (newCurrentDay != newPreviousDay) ||
                        (newCurrentWeek != newPreviousWeek) ||
                        (newCurrentMonth != newPreviousMonth) ||
                        (newCurrentYear != newPreviousYear)

    HourlyBuffer(
      totalBookings = newTotalBookings,
      distinctMonthCount = if (hasTimeChanged) buffer.distinctMonthCount + 1 else buffer.distinctMonthCount,
      previousYear = newPreviousYear,
      currentYear = newCurrentYear,
      previousMonth = newPreviousMonth,
      currentMonth = newCurrentMonth,
      previousWeek = newPreviousWeek,
      currentWeek = newCurrentWeek,
      previousDay = newPreviousDay,
      currentDay = newCurrentDay,
      previousHour = newPreviousHour,
      currentHour = newCurrentHour
    )
  }

  def finish(buffer: HourlyBuffer): Double =
    buffer.totalBookings.toDouble / buffer.distinctMonthCount

  def bufferEncoder: Encoder[HourlyBuffer] = Encoders.product
  def outputEncoder: Encoder[Double] = Encoders.scalaDouble
}