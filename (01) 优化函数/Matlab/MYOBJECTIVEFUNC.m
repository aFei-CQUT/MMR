function y = MyObjectiveFunc(x)
    y = x(1)^4 + 3*x(1)^2 + x(2)^2 - 2*x(1) - 2*x(2) - 2*x(1)^2*x(2) + 6
end