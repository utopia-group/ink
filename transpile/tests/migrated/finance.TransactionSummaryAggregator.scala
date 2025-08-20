package com.cloudera.streaming.examples.flink.operators

import org.apache.spark.sql.{Encoder, Encoders}
import org.apache.spark.sql.expressions.Aggregator

case class Transaction(
  itemId: String,
  quantity: Double
)

case class TransactionResult(
  transaction: Transaction,
  success: Boolean
)

case class TransactionSummary(
  var processed: Boolean,
  var itemId: String,
  var numSuccessfulTransactions: Long,
  var numFailedTransactions: Long,
  var totalVolume: Double
)

class TransactionSummaryAggregator extends Aggregator[TransactionResult, TransactionSummary, TransactionSummary] {
  
  def zero: TransactionSummary = TransactionSummary(false, "", 0L, 0L, 0.0)
  
  def reduce(acc: TransactionSummary, tr: TransactionResult): TransactionSummary = {
    TransactionSummary(
      true,
      tr.transaction.itemId,
      acc.numSuccessfulTransactions + (if (tr.success) 1 else 0),
      acc.numFailedTransactions + (if (!tr.success) 1 else 0),
      acc.totalVolume + math.abs(tr.transaction.quantity)
    )
  }
  
  def finish(acc: TransactionSummary): TransactionSummary = acc
  
  def bufferEncoder: Encoder[TransactionSummary] = Encoders.product
  def outputEncoder: Encoder[TransactionSummary] = Encoders.product
} 