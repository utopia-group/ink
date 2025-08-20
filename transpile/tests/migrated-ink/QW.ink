(acc: (num, num, num, num, num)) -> (bid: num) -> (acc._0 + bid, acc._1 + 1, ITE(bid < acc._2, bid, acc._2), ITE(bid > acc._3, bid, acc._3), acc._4 + bid * bid)
(0, 0, _mx, _mn, 0)
