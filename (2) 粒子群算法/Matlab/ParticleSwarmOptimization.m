function ParticleSwarmOptimization
    % 参数设置
    particle_population_size = 50;  % 粒子群大小
    max_iterations = 3000;           % 最大迭代次数
    inertia_weight = 0.7;            % 惯性权重
    individual_learning_factor = 1.5; % 个体学习因子
    social_learning_factor = 1.5;     % 社会学习因子
    dimension = 2;                    % 维度

    % 定义每个维度的边界
    x_position_bound = [-2*pi, 2*pi];  % x 位置范围
    y_position_bound = [-2*pi, 2*pi];  % y 位置范围
    position_bounds = [x_position_bound; y_position_bound];  % 组合成位置范围

    % 定义速度边界
    x_velocity_bound = [-0.5, 0.5];    % x 维度速度范围
    y_velocity_bound = [-0.5, 0.5];    % y 维度速度范围
    velocity_bounds = [x_velocity_bound; y_velocity_bound];  % 组合成速度范围
    
    % 初始化种群
    particle_positions = [position_bounds(1,1) + (position_bounds(1,2) - position_bounds(1,1)) * rand(particle_population_size, 1), ...
                          position_bounds(2,1) + (position_bounds(2,2) - position_bounds(2,1)) * rand(particle_population_size, 1)];
    particle_velocities = [velocity_bounds(1,1) + (velocity_bounds(1,2) - velocity_bounds(1,1)) * rand(particle_population_size, 1), ...
                           velocity_bounds(2,1) + (velocity_bounds(2,2) - velocity_bounds(2,1)) * rand(particle_population_size, 1)];
    particle_fitness = cellfun(@objective_function, mat2cell(particle_positions, ones(particle_population_size, 1), dimension));
    
    % 初始化个体最优和全局最优
    personal_best_positions = particle_positions;
    personal_best_fitness = particle_fitness;
    [global_best_fitness, global_best_index] = max(particle_fitness);
    global_best_position = particle_positions(global_best_index, :);
    
    % 存储每次迭代的全局最优
    global_best_history = zeros(max_iterations, 1);
    global_best_history(1) = global_best_fitness;
    
    % PSO 主循环
    for iteration = 2:max_iterations
        random_factor1 = rand(particle_population_size, dimension);
        random_factor2 = rand(particle_population_size, dimension);
        
        % 更新速度和位置
        particle_velocities = inertia_weight * particle_velocities ...
            + individual_learning_factor * random_factor1 .* (personal_best_positions - particle_positions) ...
            + social_learning_factor * random_factor2 .* (global_best_position - particle_positions);
        
        % 应用速度边界
        particle_velocities(:,1) = max(min(particle_velocities(:,1), velocity_bounds(1,2)), velocity_bounds(1,1));
        particle_velocities(:,2) = max(min(particle_velocities(:,2), velocity_bounds(2,2)), velocity_bounds(2,1));
        
        % 更新粒子位置
        particle_positions = particle_positions + particle_velocities;
        
        % 应用位置边界
        particle_positions(:,1) = max(min(particle_positions(:,1), position_bounds(1,2)), position_bounds(1,1));
        particle_positions(:,2) = max(min(particle_positions(:,2), position_bounds(2,2)), position_bounds(2,1));
        
        % 计算适应度
        particle_fitness = cellfun(@objective_function, mat2cell(particle_positions, ones(particle_population_size, 1), dimension));
        
        % 更新个体最优
        mask = particle_fitness > personal_best_fitness;
        personal_best_positions(mask, :) = particle_positions(mask, :);
        personal_best_fitness(mask) = particle_fitness(mask);
        
        % 更新全局最优
        [current_global_best_fitness, global_best_index] = max(personal_best_fitness);
        if current_global_best_fitness > global_best_fitness
            global_best_fitness = current_global_best_fitness;
            global_best_position = personal_best_positions(global_best_index, :);
        end
        
        global_best_history(iteration) = global_best_fitness;
    end
    
    % 打印结果
    disp('全局最优位置:');
    disp(global_best_position);
    disp('全局最优适应度:');
    disp(global_best_fitness);
    
    % 绘制适应度随迭代次数的变化
    figure;
    plot(global_best_history, 'LineWidth', 2);
    title('适应度随迭代次数的变化');
    xlabel('迭代次数');
    ylabel('适应度');
    grid on;
end

function fitness_value = objective_function(position)
    % 目标函数定义
    if all(position == 0)
        fitness_value = (exp((cos(2*pi*position(1)) + cos(2*pi*position(2))) / 2) - 2.71289);
    else
        fitness_value = (sin(sqrt(position(1)^2 + position(2)^2)) / sqrt(position(1)^2 + position(2)^2) + ...
                         exp((cos(2*pi*position(1)) + cos(2*pi*position(2))) / 2) - 2.71289);
    end
end