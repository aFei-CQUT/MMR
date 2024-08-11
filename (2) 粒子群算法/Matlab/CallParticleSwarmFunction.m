% 定义目标函数，这里使用 De Jong 5 函数作为示例
function f = DeJong5Function(x)
    % De Jong 5 函数的定义
    % 该函数有多个局部最小值，适合测试优化算法
    f = sum(x.^2) + sum((x(1:end-1).^2 - 10*cos(2*pi*x(1:end-1))).^2);
end

% 主程序
function optimizeDeJong5
    % 设置随机数生成器以保证结果可重复
    rng default;

    % 定义变量的数量
    nvars = 2; % 这里我们使用 2 个变量进行优化

    % 定义变量的下界和上界
    lb = [-50; -50]; % 下界
    ub = [50; 50];   % 上界

    % 设置粒子群优化的选项
    options = optimoptions('particleswarm', 'SwarmSize', 1000, 'MaxIterations', 1000);

    % 调用粒子群优化函数进行求解
    [x, fval, exitflag] = particleswarm(@DeJong5Function, nvars, lb, ub, options);

    % 输出结果
    fprintf('最优解: [%f, %f]\n', x(1), x(2));
    fprintf('最优目标函数值: %f\n', fval);
    fprintf('退出标志: %d\n', exitflag);
end

% 运行主程序
optimizeDeJong5;