with open(r"C:\Users\shute\PycharmProjects\cryptic_solver_V2\report.py", "r", encoding="utf-8") as f:
    for i, line in enumerate(f.readlines(), 1):
        if "review_with_api(" in line and "def " not in line:
            print(f"  L{i}: {line.rstrip()}")