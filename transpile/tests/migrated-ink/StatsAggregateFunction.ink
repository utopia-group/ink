(buffer: (num, set[num])) -> (value: num) -> let updatedUsers = set_add(value, buffer._1) in (buffer._0 + 1, updatedUsers)
(0, empty_set)
