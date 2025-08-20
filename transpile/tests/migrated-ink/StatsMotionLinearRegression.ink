(acc: (list[num], list[num], list[num], list[num], num, num, str)) -> (tuple: (num, num, num, num, num, str)) -> (acc._0 ++ tuple._0 :: nil, acc._1 ++ tuple._1 :: nil, acc._2 ++ tuple._2 :: nil, acc._3 ++ tuple._3 :: nil, ITE(acc._4 > tuple._3, acc._4, tuple._3), ITE(acc._5 > tuple._4, acc._5, tuple._4), tuple._5)
(nil, nil, nil, nil, -1, 0, "")
