print("Basic Calculator")

try:
    a = float(input("Mời nhập số a: "))
    b = float(input("Mời nhập số b: "))
    phep_tinh = input("Chọn phép tính (+, -, *, /): ").strip()

    if phep_tinh == "+":
        ket_qua = a + b
    elif phep_tinh == "-":
        ket_qua = a - b
    elif phep_tinh == "*":
        ket_qua = a * b
    elif phep_tinh == "/":
        ket_qua = a / b
    else:
        ket_qua = None
        print("Phép tính không hợp lệ. Hãy chọn +, -, * hoặc /.")

    if ket_qua is not None:
        print(f"Kết quả: {a:g} {phep_tinh} {b:g} = {ket_qua:g}")
except ValueError:
    print("Dữ liệu không hợp lệ. Vui lòng nhập số.")
except ZeroDivisionError:
    print("Không thể chia cho 0.")
