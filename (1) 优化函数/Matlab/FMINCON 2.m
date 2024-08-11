function example_to_call_fmincon()
    x0 = ones(1, 5);
    A = [-1, -1, -1];
    b = -100;
    Aeq = [];
    beq = [];
    lb = [];
    ub = [];
    [xopt, fopt] = fmincon(@objective_func,x0,...
                           A, b, Aeq, beq, lb,...
                           ub, non_linear_constraint_func)
    xopt
    fopt
end

function f = objective_func(x)
    f = 4*x(1) - 5*x(2) + 3* x(3) +9* x(4) -10*x(5) -12;
end

function [g, ceq] = non_linear_constraint_func(x)
    g(1)=x(3) + 2*x(4).^2 + 4*x(5) - 1200;
    g(2) = x(1)*x(3) - 3000;
    ceq =[];
end