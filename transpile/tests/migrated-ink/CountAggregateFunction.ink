(buffer: (num, str, num, num)) -> (value: (num, num)) -> let newTimestamp = ITE(value._0 > buffer._2, value._0, buffer._2) in let newStimulus = ITE(value._1 > buffer._3, value._1, buffer._3) in (buffer._0 + 1, buffer._1, newTimestamp, newStimulus)
(0, "", _mn, _mn)
