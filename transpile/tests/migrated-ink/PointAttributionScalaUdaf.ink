(buffer: num) -> (input: num) -> let MAX_POINT_PER_ORDER = 3 in let points = ITE(input < MAX_POINT_PER_ORDER, input, MAX_POINT_PER_ORDER) in buffer + points
0
