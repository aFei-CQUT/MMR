% 目标函数的系数矩阵
f = [7, 5, 9, 6, 3]
% 指定索引的矩阵
intcon = [1, 2, 3, 4, 5]
% 线性约束等式系数矩阵
A = [
    56, 20, 54, 42, 15;
    -1, -2, 0, -1, -2;
    1, 4, 1, 0, 0;
]
% 线性约束的常数项向量矩阵
b = [
    100;
    4;
    2
]
% 下界
lb = zeros(5, 1)
% 上界
ub = ones(5, 1)


% 调用
[xopt, fopt] = intlinprog(f, intcon, A, b, lb, ub)
% 全格式 [xopt, fopt] = intlinprog(f, intcon, A, b, Aeq,...
                                 beq, lb, ub, x0, options)