function OptimizeDeJong5
    % ����������������Ա�֤������ظ�
    rng default

    % �������������
    nvars = 2;

    % ����������½�
    lb = [-50, -50];

    % ����������Ͻ�
    ub = [+50, +50];

    % ��������Ⱥ�Ż����Ż�ѡ��
    options = optimoptions('particleswarm', 'SwarmSize', 1000,...
                            'MaxIterations', 1000);

    % ��������Ⱥ�Ż������������
    [xopt, fopt, exitflag] = particleswarm(@ObjectiveFunction,...
                                            nvars, lb, ub, options);

    % ������
    fprintf('���Ž�:[%f , %f]\n', xopt(1), xopt(2));
    fprintf('����ֵ: %f\n', fopt);
    fprintf('�˳���־: %d\n', exitflag);
end

function f = ObjectiveFunction(x)
    f = sum(x.^2) + sum( ( x.^2 - 10*cos(2*pi*x) ).^2);
end

% ����������
OptimizeDeJong5;