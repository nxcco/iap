from precision import fp16, fp8, fp4


def main():
    x = 1.24534543534534
    print(f"Original: {x}")
    print(f"FP16: {fp16(x)}")
    print(f"FP8:  {fp8(x)}")
    print(f"FP4:  {fp4(x)}")
    
    x = 34.4325
    print(f"Original: {x}")
    print(f"FP16: {fp16(x)}")
    print(f"FP8:  {fp8(x)}")
    print(f"FP4:  {fp4(x)}")
    
    x = 3.24534543534534
    print(f"Original: {x}")
    print(f"FP16: {fp16(x)}")
    print(f"FP8:  {fp8(x)}")
    print(f"FP4:  {fp4(x)}")

main()