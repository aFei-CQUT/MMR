% min -x(1) - x(2)
obj_func = [-1, -1]

% 整数约束, IntConstraints
intcon = [1, 2]

% 线性不等式约束系数矩阵, Linear Inequality Constraints
A = [
    -4, 2;
    4, 2
]

% 线性不等式约束常数项矩阵, Linear Inequality Constraints
b = [
    -1;
    11
]

% 线性等式约束系数矩阵, Linear Equality Constraints
Aeq = []

% 线性等式约束常数项矩阵, Linear Equality Constraints
beq = []

% 下限, Lower Bound
lb = zeros(2, 1) % zeros(行数, 列数) 返回 (行数*列数)的零矩阵

% 上限, Upper Bound
up = []

% 初始猜测值x0
x0 = []

% 可选配置
options = []

% 函数调用
[xopt, fopt] = intlinprog(f, intcon, A, b, Aeq, beq, lb, ub, x0, options)