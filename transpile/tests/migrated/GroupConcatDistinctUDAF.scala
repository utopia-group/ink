import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.sql.expressions.Aggregator

case class CityInfoInput(cityInfo: String)
case class CityInfoBuffer(bufferCityInfo: String)

class GroupConcatDistinctUDAF extends Aggregator[CityInfoInput, CityInfoBuffer, String] {
  
  def zero: CityInfoBuffer = CityInfoBuffer("")
  
  def reduce(buffer: CityInfoBuffer, input: CityInfoInput): CityInfoBuffer = {
    val bufferInfo = buffer.bufferCityInfo
    val inputStr = input.cityInfo
    
    if (!bufferInfo.contains(inputStr)) {
      val newBuffer = if (bufferInfo.isEmpty) inputStr else s"$bufferInfo,$inputStr"
      CityInfoBuffer(newBuffer)
    } else {
      buffer
    }
  }
  
  def merge(b1: CityInfoBuffer, b2: CityInfoBuffer): CityInfoBuffer = {
    throw new UnsupportedOperationException("merge not implemented")
  }
  
  def finish(reduction: CityInfoBuffer): String = reduction.bufferCityInfo
  
  def bufferEncoder: Encoder[CityInfoBuffer] = Encoders.product
  def outputEncoder: Encoder[String] = Encoders.STRING
} 