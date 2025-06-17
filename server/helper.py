
def encode_wheels(x, y):
    sign1 = '-' if x < 0 else '+'
    x = str(abs(x)).zfill(3)
    sign2 = '-' if y < 0 else '+'
    y = str(abs(y)).zfill(3)
    return f"{x}{sign1}{y}{sign2}"

def clamp(x, min_val=-1, max_val=1):
    return max(min_val, min(max_val, x))