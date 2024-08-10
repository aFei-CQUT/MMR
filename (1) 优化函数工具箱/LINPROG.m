% max 2*x(1) + 4*x(2) + 3*x(3)
obj_func = [-2, -4, -3]
A = [
3, 4, 2;
2, 1, 2;
1, 3, 2
]
b = [
600;
400;
800
]
Aeq = []
beq = []
lb = zeros(3, 1)
ub = []

% [xopt, fopt] = linprog(objective_func, A, ...
%                        b, Aeq, beq, ...
%                        lb, ub, x0, options)

[xopt, fopt] = linprog(obj_func, A, b, Aeq, beq, lb)