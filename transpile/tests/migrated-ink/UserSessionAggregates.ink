(buffer: (num, num)) -> (value: num) -> let newCheckoutEvents = ITE(not (value = 0), buffer._1 + 1, buffer._1) in (buffer._0 + 1, newCheckoutEvents)
(0, 0)
