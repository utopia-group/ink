(buffer: (num, num, bool)) -> (input: num) -> ITE(input = 0 || buffer._2, (buffer._0, buffer._1, true), (buffer._0 + 1, buffer._1 + 1 / input, buffer._2))
(0, 0, false)
