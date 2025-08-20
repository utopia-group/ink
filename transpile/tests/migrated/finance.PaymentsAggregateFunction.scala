import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.sql.expressions.Aggregator
import java.time.Instant

case class Payment(status: String, amount: Double)
case class PaymentContextInformation(merchantId: String, payment: Payment)

case class PaymentsAggregate(
  merchantId: String = "",
  approved: Int = 0,
  declined: Int = 0,
  canceled: Int = 0,
  others: Int = 0,
  approvedAmount: Double = 0.0,
  declinedAmount: Double = 0.0,
  canceledAmount: Double = 0.0,
  othersAmount: Double = 0.0,
  windowTimestamp: Option[Instant] = None
)

object PaymentStatus extends Enumeration {
  type PaymentStatusEnum = Value
  val APPROVED, DECLINED, CANCELLED = Value
  
  def valueOf(status: String): PaymentStatusEnum = status.toUpperCase match {
    case "APPROVED" => APPROVED
    case "DECLINED" => DECLINED
    case "CANCELLED" => CANCELLED
    case _ => DECLINED
  }
}

class PaymentsAggregateFunction extends Aggregator[PaymentContextInformation, PaymentsAggregate, PaymentsAggregate] {
  
  def zero: PaymentsAggregate = PaymentsAggregate()
  
  def reduce(buffer: PaymentsAggregate, payment: PaymentContextInformation): PaymentsAggregate = {
    val status = PaymentStatus.valueOf(payment.payment.status)
    val amount = payment.payment.amount
    
    val (newApproved, newDeclined, newCanceled, newOthers) = status match {
      case PaymentStatus.APPROVED => (buffer.approved + 1, buffer.declined, buffer.canceled, buffer.others)
      case PaymentStatus.DECLINED => (buffer.approved, buffer.declined + 1, buffer.canceled, buffer.others)
      case PaymentStatus.CANCELLED => (buffer.approved, buffer.declined, buffer.canceled + 1, buffer.others)
      case _ => (buffer.approved, buffer.declined, buffer.canceled, buffer.others + 1)
    }
    
    val (newApprovedAmount, newDeclinedAmount, newCanceledAmount, newOthersAmount) = status match {
      case PaymentStatus.APPROVED => (buffer.approvedAmount + amount, buffer.declinedAmount, buffer.canceledAmount, buffer.othersAmount)
      case PaymentStatus.DECLINED => (buffer.approvedAmount, buffer.declinedAmount + amount, buffer.canceledAmount, buffer.othersAmount)
      case PaymentStatus.CANCELLED => (buffer.approvedAmount, buffer.declinedAmount, buffer.canceledAmount + amount, buffer.othersAmount)
      case _ => (buffer.approvedAmount, buffer.declinedAmount, buffer.canceledAmount, buffer.othersAmount + amount)
    }
    
    PaymentsAggregate(
      payment.merchantId,
      newApproved,
      newDeclined,
      newCanceled,
      newOthers,
      newApprovedAmount,
      newDeclinedAmount,
      newCanceledAmount,
      newOthersAmount,
      Some(Instant.now())
    )
  }
  
  def merge(b1: PaymentsAggregate, b2: PaymentsAggregate): PaymentsAggregate = {
    throw new UnsupportedOperationException("merge not implemented")
  }
  
  def finish(reduction: PaymentsAggregate): PaymentsAggregate = reduction
  
  def bufferEncoder: Encoder[PaymentsAggregate] = Encoders.product
  def outputEncoder: Encoder[PaymentsAggregate] = Encoders.product
} 