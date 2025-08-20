(buffer: ((num, num), num)) -> (input: (num, num)) -> let newSum = (buffer._0._0 + input._0, buffer._0._1 + input._1) in (newSum, buffer._1 + 1)
((0, 0), 0)
