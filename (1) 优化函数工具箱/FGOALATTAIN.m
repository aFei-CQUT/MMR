function exmaple_to_call_fgoalattain()
    % goals其中的值为f(1),f(2),f(3)各自函数值
    goals = [200, -100, -50];
    weights = 0.05*[200, -100, -50];
    x0 = [55, 55];
    A = [
        2, 1;
        -1, -1;
        -1, 0
        ];
    b = [200, -100, -50];
    Aeq = [];
    beq = [];
    lb = zeros(2, 1);
    ub =  [];
    [xopt, fopt,attainfactor, exitflag] = ...
                fgoalattain(@objective_func, x0,...
                goal, weight, A, b, Aeq, beq,lb ,ub);
    xopt
    fopt
    attainfactor
    exitflag
end

function f = objective_func(x)
    f(1)=2*x(1)+ x(2); % f(1)表示的是原料采购总费用 
    f(2)=-x(1)- x(2); % f(2)表示的是采购总重量 
    f(3)=-x(1); % f(3)甲级原料的总质量
end
