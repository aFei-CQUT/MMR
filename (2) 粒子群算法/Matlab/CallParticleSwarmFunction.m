function OptimizeDeJong5
    % 设置随机数生成器以保证结果可重复
    rng default

    % 定义变量的数量
    nvars = 2;

    % 定义变量的下界
    lb = [-50, -50];

    % 定义变量的上界
    ub = [+50, +50];

    % 设置粒子群优化器优化选项
    options = optimoptions('particleswarm', 'SwarmSize', 1000,...
                            'MaxIterations', 1000);

    % 调用粒子群优化函数进行求解
    [xopt, fopt, exitflag] = particleswarm(@ObjectiveFunction,...
                                            nvars, lb, ub, options);

    % 输出结果
    fprintf('最优解:[%f , %f]\n', xopt(1), xopt(2));
    fprintf('最优值: %f\n', fopt);
    fprintf('退出标志: %d\n', exitflag);
end

function f = ObjectiveFunction(x)
    f = sum(x.^2) + sum( ( x.^2 - 10*cos(2*pi*x) ).^2);
end

% 运行主程序
OptimizeDeJong5;