(buffer: num) -> (input: (num, num)) -> let westernState = not (input._1 < 10) && not (input._1 > 19) in let sales = input._0 in ITE(westernState && sales > 1000 || (not westernState) && sales > 400, buffer + sales, buffer)
0
