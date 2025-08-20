(buffer: (num, num, num, num, num)) -> (input: (num, num)) -> let x = (input._0 - 2000) * 12 + 1 in let p = input._1 in (buffer._0 + x, buffer._1 + p, buffer._2 + x * p, buffer._3 + x * x, buffer._4 + 1)
(0, 0, 0, 0, 0)
