a = 'abc'
name = [k for k, v in locals().items() if v is a][0]
print(f'{name}')

