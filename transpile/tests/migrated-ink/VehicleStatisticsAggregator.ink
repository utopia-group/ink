(buffer: (str, num, num, num)) -> (vehicle: (str, num)) -> let vehicleType = ITE(buffer._0 = "", vehicle._0, buffer._0) in let start = ITE(buffer._1 = 0, vehicle._1, buffer._1) in let count = ITE(buffer._3 = 0, 1, buffer._3 + 1) in (vehicleType, start, vehicle._1, count)
("", 0, 0, 0)
