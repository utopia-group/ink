(buffer: map[str, num]) -> (input: map[str, num]) -> concat_map(filter_values((value: num) -> not (value = 0), input), buffer)
empty_map
