import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.sql.expressions.Aggregator
import org.apache.spark.sql.types._
import java.math.{BigDecimal => JBigDecimal}
import java.sql.Timestamp
import java.time.LocalDateTime

case class TickInput(
  day: String,
  hour: Int,
  min: Int,
  sec: Int,
  milli: Int,
  tradHour: Int,
  tradMin: Int,
  price: JBigDecimal,
  thresh: Int
)

case class PriceBuffer(price: Option[JBigDecimal], priceTs: Option[Timestamp])
case class PriceResult(price: Option[JBigDecimal], priceTs: Option[Timestamp])

abstract class BaseAggregator(colName: String) extends Aggregator[TickInput, PriceBuffer, PriceResult] {
  
  def zero: PriceBuffer = PriceBuffer(None, None)
  
  def reduce(buffer: PriceBuffer, input: TickInput): PriceBuffer = {
    val nfpTs = getNfpTs(input.day, input.tradHour, input.tradMin)
    val tickTs = getTickTs(input.day, input.hour, input.min, input.sec, input.milli)
    
    if (tickTs.after(nfpTs) && tickTs.before(tsPlusMins(nfpTs, input.thresh))) {
      PriceBuffer(Some(input.price), Some(tickTs))
    } else {
      buffer
    }
  }
  
  def merge(b1: PriceBuffer, b2: PriceBuffer): PriceBuffer = {
    throw new UnsupportedOperationException("merge not implemented")
  }
  
  def finish(reduction: PriceBuffer): PriceResult = {
    PriceResult(reduction.price, reduction.priceTs)
  }
  
  def bufferEncoder: Encoder[PriceBuffer] = Encoders.product
  def outputEncoder: Encoder[PriceResult] = Encoders.product
  
  private def getTickTs(day: String, hour: Int, min: Int, sec: Int, milli: Int): Timestamp = {
    val localDateTime = LocalDateTime.of(
      day.substring(0, 4).toInt,
      day.substring(4, 6).toInt,
      day.substring(6, 8).toInt,
      hour, min, sec, milli * 1000000
    )
    Timestamp.valueOf(localDateTime)
  }
  
  private def getNfpTs(day: String, nfpHour: Int, nfpMin: Int): Timestamp = {
    val localDateTime = LocalDateTime.of(
      day.substring(0, 4).toInt,
      day.substring(4, 6).toInt,
      day.substring(6, 8).toInt,
      nfpHour, nfpMin, 0, 0
    )
    Timestamp.valueOf(localDateTime)
  }
  
  private def tsPlusMins(ts: Timestamp, thresh: Int): Timestamp = {
    val localDateTime = ts.toLocalDateTime.plusMinutes(thresh)
    Timestamp.valueOf(localDateTime)
  }
}

class OpenAggregator(colName: String) extends BaseAggregator(colName) 