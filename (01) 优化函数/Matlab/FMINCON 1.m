function example_to_call_fmincon()
    x0 = [2; 4; 5];
    lb = [];
    ub = [];
    Aeq = [1, 0, 1];
    beq = 7;
    [xopt, fopt] = fmincon(@objective_func, x0,...
                          [], [], Aeq, beq, lb,...
                          ub, @non_linear_constraint_func);
    xopt
    fopt
end

function f = objective_func(x)
    f = 4*x(1) - x(2).^2 + x(3).^2 -12;
end

function [g, ceq] = non_linear_constraint_func(x)
    ceq = x(1).^2 + x(2).^2 - 20;
    g = [];
end