with open('listen (4).pls') as file:
    for line in file:
        if 'http' in line:
            print(line.split('=')[1])