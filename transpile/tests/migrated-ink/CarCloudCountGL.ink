(acc: (num, num, num, num)) -> (input: (num, num, num, num, num)) -> let newTimestamp = ITE(input._4 > acc._3, input._4, acc._3) in (input._0, acc._1 + 1, ITE(input._3 > acc._2, input._3, acc._2), newTimestamp)
(0, 0, _mn, _mn)
