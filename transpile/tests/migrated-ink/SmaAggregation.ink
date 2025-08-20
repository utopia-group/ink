(buffer: ((num, num, num), num)) -> (value: (num, num, num)) -> let newStartTime = ITE(value._0 < buffer._0._0, value._0, buffer._0._0) in let newEndTime = ITE(value._1 > buffer._0._1, value._1, buffer._0._1) in let newPayload = buffer._0._2 + value._2 in let newCount = buffer._1 + 1 in ((newStartTime, newEndTime, newPayload), newCount)
((_mx, _mn, 0), 0)
