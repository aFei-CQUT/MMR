% 求解二维无约束优化问题
% fminsearch 调用格式
% [xopt , fopt] = fminsearch(func, x0<该参为初始点>, options)

% 内联函数的形式
% func = inline('x(1)^4+3*x(1)^2+x(2^2-2*x(1)-2*x(2)-2*x(1)^2*x(2)+6')
% 匿名函数的形式
% func = @x xx(1)^4+3*x(1)^2+x(2^2-2*x(1)-2*x(2)-2*x(1)^2*x(2)+6
% 字符串的形式,然后用feval动态调用函数
func = 'x(1)^4+3*x(1)^2+x(2^2-2*x(1)-2*x(2)-2*x(1)^2*x(2)+6'

x0 = [0, 0];
[xopt, fopt] = fminsearch(@(x) feval(func), x0)