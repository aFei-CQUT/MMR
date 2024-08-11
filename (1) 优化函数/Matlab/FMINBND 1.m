% 求解一维无约束优化问题

% 内联函数的用法
func = inline("(x^3+cos(x)+x*log(x))/exp(x)","x");
% 搜索区间左边界
x1 = 0;
% 搜索区间右边界
x2 = 1;

[xopt, fopt] = fminbnd(func, x1, x2)
title("(x^3+cos(x)+x*log(x))/exp(x)")
grid on