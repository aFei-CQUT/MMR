function example_fminunc()
    x0 = [25, 45];  % 初始点
    options = optimoptions('fminunc', 'Display', 'iter', 'Algorithm', 'quasi-newton');
    [x, Fmin] = fminunc(@FuncToOptimize, x0, options);  % 求优语句
    fprintf(1, '截面高度 h     x(1) = %3.4f mm\n', x(1));
    fprintf(1, '斜边夹角 θ     x(2) = %3.4f 度\n', x(2));
    fprintf(1, '截面周长 s     f    = %3.4f mm\n', Fmin);
    plot_figure();
end

function f = FuncToOptimize(x)  % 定义目标函数调用格式
    a = 64516; 
    θ = pi / 180;  % 将角度转换为弧度
    f = a / x(1) - x(1) / tan(x(2) * θ) + 2 * x(1) / sin(x(2) * θ);  % 定义目标函数
end

function plot_figure()
    xx1 = linspace(100, 300, 25);   % 高度范围
    xx2 = linspace(30, 120, 25);    % 角度范围
    [x1, x2] = meshgrid(xx1, xx2);
    a = 64516; 
    θ = pi / 180;  % 将角度转换为弧度
    f = a ./ x1 - x1 ./ tan(x2 * θ) + 2 * x1 ./ sin(x2 * θ);
    
    subplot(1, 2, 1);
    h = contour(x1, x2, f);
    clabel(h);
    axis([100, 300, 30, 120]);
    xlabel('高度 h/mm');
    ylabel('倾斜角 θ/(°)');
    title('目标函数等值线');
    
    subplot(1, 2, 2);
    meshc(x1, x2, f);
    axis([100, 300, 30, 120, 600, 1200]);
    title('目标函数网格曲面图');
end