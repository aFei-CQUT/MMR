% ����Ŀ�꺯��������ʹ�� De Jong 5 ������Ϊʾ��
function f = DeJong5Function(x)
    % De Jong 5 �����Ķ���
    % �ú����ж���ֲ���Сֵ���ʺϲ����Ż��㷨
    f = sum(x.^2) + sum((x(1:end-1).^2 - 10*cos(2*pi*x(1:end-1))).^2);
end

% ������
function optimizeDeJong5
    % ����������������Ա�֤������ظ�
    rng default;

    % �������������
    nvars = 2; % ��������ʹ�� 2 �����������Ż�

    % ����������½���Ͻ�
    lb = [-50; -50]; % �½�
    ub = [50; 50];   % �Ͻ�

    % ��������Ⱥ�Ż���ѡ��
    options = optimoptions('particleswarm', 'SwarmSize', 1000, 'MaxIterations', 1000);

    % ��������Ⱥ�Ż������������
    [x, fval, exitflag] = particleswarm(@DeJong5Function, nvars, lb, ub, options);

    % ������
    fprintf('���Ž�: [%f, %f]\n', x(1), x(2));
    fprintf('����Ŀ�꺯��ֵ: %f\n', fval);
    fprintf('�˳���־: %d\n', exitflag);
end

% ����������
optimizeDeJong5;