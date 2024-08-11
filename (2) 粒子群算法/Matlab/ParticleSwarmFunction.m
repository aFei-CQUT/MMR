function ParticleSwarmFunction

% 搜索空间的维度
Dimension = 40;

% 种群规模
PopulationSize = 100;

% 最大迭代次数
MaxIterations = 10000;

% 认知学习因子
IndividualLearningFactor = 2;

% 社会学习因子
SocialLearningFactor = 2;

% 选择不同测试函数的速度和位置限制范围
FunctionNumber = 2; % 选择测试函数编号

switch FunctionNumber
    case 1 % f1_Sphere [-30, 30]
        VelocityMax(1:Dimension) = 30;
        VelocityMin(1:Dimension) = -30;
        PositionMax(1:Dimension) = 30;
        PositionMin(1:Dimension) = -30;
    case 2 % f2_Griewank [-600,600]
        VelocityMax(1:Dimension) = 600;
        VelocityMin(1:Dimension) = -600;
        PositionMax(1:Dimension) = 600;
        PositionMin(1:Dimension) = -600;
    case 3 % f3_Rastrigin [-5.12,5.12]
        VelocityMax(1:Dimension) = 5.12;
        VelocityMin(1:Dimension) = -5.12;
        PositionMax(1:Dimension) = 5.12;
        PositionMin(1:Dimension) = -5.12;
    case 4 % f4_Rosenbrock [-2.408,2.408]
        VelocityMax(1:Dimension) = 2.408;
        VelocityMin(1:Dimension) = -2.408;
        PositionMax(1:Dimension) = 2.408;
        PositionMin(1:Dimension) = -2.408;
end

% 初始化粒子的位置和速度
[ParticlePositions, ParticleVelocities] = InitializeParticles(...
    Dimension, PopulationSize, PositionMax, PositionMin, ...
    VelocityMax, VelocityMin);

% 初始化个体最优位置
PersonalBests = ParticlePositions;

% 评估每个粒子的适应度并找到全局最优
for ParticleIndex = 1:PopulationSize
    ParticlePosition = ParticlePositions(:, ParticleIndex);
    ParticleFitness(ParticleIndex) = FitnessFunction(ParticlePosition, FunctionNumber);
end

[GlobalBestFitness, GlobalBestIndex] = min(ParticleFitness);
GlobalBest = ParticlePositions(:, GlobalBestIndex);

% 迭代更新粒子的位置和速度
for Iteration = 1:MaxIterations
    IterationNumber(Iteration) = Iteration;

    % 惯性权重
    InertiaWeight = 1;

    % 生成随机数
    CognitiveRandom = rand(1);
    SocialRandom = rand(1);

    % 更新速度
    for ParticleIndex = 1:PopulationSize
        ParticleVelocities(:, ParticleIndex) = ...
            InertiaWeight * ParticleVelocities(:, ParticleIndex) + ...
            IndividualLearningFactor * CognitiveRandom * ...
            (PersonalBests(:, ParticleIndex) - ParticlePositions(:, ParticleIndex)) + ...
            SocialLearningFactor * SocialRandom * ...
            (GlobalBest - ParticlePositions(:, ParticleIndex));
    end

    % 限制速度
    for ParticleIndex = 1:PopulationSize
        VelocityLimitIndices = 1;
        VelocityLimitFactors = ones(Dimension, 1);
        for DimensionIndex = 1:Dimension
            if ParticleVelocities(DimensionIndex, ParticleIndex) > VelocityMax(DimensionIndex)
                VelocityLimitFactors(VelocityLimitIndices) = ...
                    VelocityMax(DimensionIndex) / ParticleVelocities(DimensionIndex, ParticleIndex);
                VelocityLimitIndices = VelocityLimitIndices + 1;
            elseif ParticleVelocities(DimensionIndex, ParticleIndex) < VelocityMin(DimensionIndex)
                VelocityLimitFactors(VelocityLimitIndices) = ...
                    VelocityMin(DimensionIndex) / ParticleVelocities(DimensionIndex, ParticleIndex);
                VelocityLimitIndices = VelocityLimitIndices + 1;
            end
        end
        VelocityLimitFactor = min(VelocityLimitFactors);
        for DimensionIndex = 1:Dimension
            if ParticleVelocities(DimensionIndex, ParticleIndex) > VelocityMax(DimensionIndex)
                ParticleVelocities(DimensionIndex, ParticleIndex) = ...
                    ParticleVelocities(DimensionIndex, ParticleIndex) * VelocityLimitFactor;
            elseif ParticleVelocities(DimensionIndex, ParticleIndex) < VelocityMin(DimensionIndex)
                ParticleVelocities(DimensionIndex, ParticleIndex) = ...
                    ParticleVelocities(DimensionIndex, ParticleIndex) * VelocityLimitFactor;
            end
        end
    end

    % 更新位置
    ParticlePositions = ParticlePositions + ParticleVelocities;

    % 限制位置
    for ParticleIndex = 1:PopulationSize
        for DimensionIndex = 1:Dimension
            if ParticlePositions(DimensionIndex, ParticleIndex) > PositionMax(DimensionIndex)
                ParticlePositions(DimensionIndex, ParticleIndex) = PositionMax(DimensionIndex);
            elseif ParticlePositions(DimensionIndex, ParticleIndex) < PositionMin(DimensionIndex)
                ParticlePositions(DimensionIndex, ParticleIndex) = PositionMin(DimensionIndex);
            end
        end
    end

    % 重新评估适应度，更新个体最优和全局最优
    for ParticleIndex = 1:PopulationSize
        ParticlePosition = ParticlePositions(:, ParticleIndex)';
        ParticleNewFitness = FitnessFunction(ParticlePosition, FunctionNumber);

        if ParticleNewFitness < ParticleFitness(ParticleIndex)
            PersonalBests(:, ParticleIndex) = ParticlePositions(:, ParticleIndex);
            ParticleFitness(ParticleIndex) = ParticleNewFitness;
        end

        if ParticleNewFitness < GlobalBestFitness
            GlobalBestFitness = ParticleNewFitness;
        end
    end

    [GlobalBestFitness, GlobalBestIndex] = min(ParticleFitness);
    GlobalBestFitnessHistory(Iteration) = GlobalBestFitness;
    GlobalBest = PersonalBests(:, GlobalBestIndex);
end

% 绘制全局最优适应度变化曲线
figure(1);
plot(IterationNumber, GlobalBestFitnessHistory, '-b');
legend('优化后的 PSO');
xlabel('迭代次数');
ylabel('全局适应度');
hold on;

end

function [ParticlePositions, ParticleVelocities] = InitializeParticles(...
    Dimension, PopulationSize, PositionMax, PositionMin, ...
    VelocityMax, VelocityMin)

% 初始化粒子的位置和速度
for DimensionIndex = 1:Dimension
    ParticlePositions(DimensionIndex, :) = ...
        PositionMin(DimensionIndex) + ...
        (PositionMax(DimensionIndex) - PositionMin(DimensionIndex)) * rand(1, PopulationSize);
    ParticleVelocities(DimensionIndex, :) = ...
        VelocityMin(DimensionIndex) + ...
        (VelocityMax(DimensionIndex) - VelocityMin(DimensionIndex)) * rand(1, PopulationSize);
end

end

function Fitness = FitnessFunction(ParticlePosition, FunctionNumber)

% 计算粒子的适应度
Dimension = size(ParticlePosition, 1);

% 选择标准测试函数
switch FunctionNumber
    case 1
        % f1_Sphere
        SphereFunction = ParticlePosition(:)' * ParticlePosition(:);
        Fitness = SphereFunction;
    case 2
        % f2_Griewank
        GriewankTerm1 = ParticlePosition(:)' * ParticlePosition(:) / 4000;
        GriewankTerm2 = 1;
        for DimensionIndex = 1:Dimension
            GriewankTerm2 = GriewankTerm2 * cos(ParticlePosition(DimensionIndex) / sqrt(DimensionIndex));
        end
        GriewankFunction = GriewankTerm1 - GriewankTerm2 + 1;
        Fitness = GriewankFunction;
    case 3
        % f3_Rastrigin
        RastriginFunction = ParticlePosition(:)' * ParticlePosition(:) - ...
            10 * sum(cos(ParticlePosition(:) * 2 * pi)) + 10 * Dimension;
        Fitness = RastriginFunction;
    case 4
        % f4_Rosenbrock
        RosenbrockTerm = 0;
        for DimensionIndex = 1:(Dimension - 1)
            RosenbrockTerm = RosenbrockTerm + ...
                100 * (ParticlePosition(DimensionIndex + 1) - ...
                ParticlePosition(DimensionIndex)^2)^2 + ...
                (ParticlePosition(DimensionIndex) - 1)^2;
        end
        Fitness = RosenbrockTerm;
end

end