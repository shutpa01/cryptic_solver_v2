with open(r"C:\Users\shute\PycharmProjects\cryptic_solver_V2\stages\general.py", "r", encoding="utf-8") as f:
    lines = f.readlines()
    for i, line in enumerate(lines, 1):
        if "def main" in line or "__name__" in line:
            print(f"  L{i}: {line.rstrip()}")