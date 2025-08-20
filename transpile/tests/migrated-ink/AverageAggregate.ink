(buffer: (str, num, num, num, num)) -> (bus: (str, num)) -> let minSpeed = ITE(bus._1 < buffer._2, bus._1, buffer._2) in let maxSpeed = ITE(bus._1 > buffer._3, bus._1, buffer._3) in (bus._0, buffer._1 + bus._1, minSpeed, maxSpeed, buffer._4 + 1)
("", 0, _mx, _mn, 0)
