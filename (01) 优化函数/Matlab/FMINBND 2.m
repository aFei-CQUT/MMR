% 求解一维无约束优化函数

function main()
    x1 = 0;
    x2 = 4;
    func = @f1;
    % 调用fminbnd优化函数
    [xpot, fopt] = fminbnd(func, x1, x2);
    
    % 作图
    x = x1:0.001:x2;  % 小步长，密集点，曲线光滑
    y = func(x);
    plot(x, y)
    title('$\frac{x^3+\cos(x)+x\ln(x)}{\exp(x)}$', 'Interpreter', 'latex')
    xlabel('x')
    ylabel('y')
    grid on
end

% 定义目标函数作为局部函数的用法
% 当项目庞大时，应作为函数文件单独保存
function y = f1(x)
    y = (x.^3 + cos(x) + x .* log(x)) ./ exp(x);
end