import sympy as sp

# 定义符号变量
x = sp.symbols('x')
y1 = sp.Function('y1')(x)
y2 = sp.Function('y2')(x)

# 定义二阶微分方程
ode1 = sp.Eq(y1.diff(x, 2) + y1, 0)
ode2 = sp.Eq(y2.diff(x, 2) - y2, 0)

# 分别求解每个方程
solution_y1 = sp.dsolve(ode1)
solution_y2 = sp.dsolve(ode2)

# 方程组的解
combined_solution = {
    'y1': solution_y1,
    'y2': solution_y2
}

print("\n该二阶微分方程组的解:")
for key, sol in combined_solution.items():
    print(f"{key}: {sol}")

# 打印LaTeX格式的输出
latex_output_y1 = sp.latex(solution_y1)
latex_output_y2 = sp.latex(solution_y2)

print("\n第一个方程解的LaTeX格式输出:")
print(latex_output_y1)

print("\n第二个方程解的LaTeX格式输出:")
print(latex_output_y2)
