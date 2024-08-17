function example_fminunc()
    x0 = [25, 45];
    options = optimoptions('fminunc', 'Display', 'iter', 'Algorithm', 'quasi-newton')
