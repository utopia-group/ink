(acc: (num, num)) -> (value: num) -> let newOutlierCount = ITE(value > 100, acc._1 + 1, acc._1) in (acc._0 + 1, newOutlierCount)
(0, 0)
