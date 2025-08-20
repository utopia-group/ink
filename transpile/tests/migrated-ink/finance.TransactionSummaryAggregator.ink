(acc: (bool, str, num, num, num)) -> (tr: ((str, num), bool)) -> (true, tr._0._0, acc._2 + ITE(tr._1, 1, 0), acc._3 + ITE((not tr._1), 1, 0), acc._4 + abs(tr._0._1))
(false, "", 0, 0, 0)
