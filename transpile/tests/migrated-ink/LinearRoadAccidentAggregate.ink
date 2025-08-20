(acc: (set[num], num, num, num, num, num, num, num)) -> (tuple: (num, num, num, num, num, num, num, num, bool, num, str)) -> let newVids = set_add(tuple._1, acc._0) in (newVids, ITE(acc._1 > tuple._0, acc._1, tuple._0), ITE(acc._2 > tuple._9, acc._2, tuple._9), tuple._3, tuple._4, tuple._5, tuple._6, tuple._7)
(empty_set, _mn, _mn, 0, 0, 0, 0, 0)
