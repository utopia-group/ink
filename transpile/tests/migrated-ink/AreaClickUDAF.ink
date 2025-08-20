(buffer: (map[str, num], num)) -> (input: str) -> let updatedMap = buffer._0[input <- buffer._0[input] + 1] in (updatedMap, buffer._1 + 1)
(empty_map, 0)
