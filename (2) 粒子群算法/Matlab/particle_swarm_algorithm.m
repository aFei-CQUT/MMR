function particle_swarm_algorithm
    % 搜索空间的维度
    dimension = 40;
    
    % 种群规模
    population_size = 100;
    
    % 最大迭代次数
    max_iterations = 10000;
    
    % 认知学习因子
    individual_learning_factor = 2;
    
    % 社会学习因子
    social_learning_factor = 2;
    
    % 选择不同测试函数的速度和位置限制范围
    function_number = 2; % 选择测试函数编号
    
    switch function_number
        case 1 % f1_Sphere [-30, 30]
            velocity_max(1:dimension) = 30;
            velocity_min(1:dimension) = -30;
            position_max(1:dimension) = 30;
            position_min(1:dimension) = -30;
        case 2 % f2_Griewank [-600,600]
            velocity_max(1:dimension) = 600;
            velocity_min(1:dimension) = -600;
            position_max(1:dimension) = 600;
            position_min(1:dimension) = -600;
        case 3 % f3_Rastrigin [-5.12,5.12]
            velocity_max(1:dimension) = 5.12;
            velocity_min(1:dimension) = -5.12;
            position_max(1:dimension) = 5.12;
            position_min(1:dimension) = -5.12;
        case 4 % f4_Rosenbrock [-2.408,2.408]
            velocity_max(1:dimension) = 2.408;
            velocity_min(1:dimension) = -2.408;
            position_max(1:dimension) = 2.408;
            position_min(1:dimension) = -2.408;
    end
    
    % 初始化粒子的位置和速度
    [particle_positions, particle_velocities] = initialize_particles(...
        dimension, population_size, position_max, position_min, ...
        velocity_max, velocity_min);
    
    % 初始化个体最优位置
    personal_bests = particle_positions;
    
    % 评估每个粒子的适应度并找到全局最优
    for particle_index = 1:population_size
        particle_position = particle_positions(:, particle_index);
        particle_fitness(particle_index) = fitness_function(particle_position, function_number);
    end
    
    [global_best_fitness, global_best_index] = min(particle_fitness);
    global_best = particle_positions(:, global_best_index);
    
    % 迭代更新粒子的位置和速度
    for iteration = 1:max_iterations
        iteration_number(iteration) = iteration;
        
        % 惯性权重
        inertia_weight = 1;
        
        % 生成随机数
        cognitive_random = rand(1);
        social_random = rand(1);
        
        % 更新速度
        for particle_index = 1:population_size
            particle_velocities(:, particle_index) = ...
                inertia_weight * particle_velocities(:, particle_index) + ...
                individual_learning_factor * cognitive_random * ...
                (personal_bests(:, particle_index) - particle_positions(:, particle_index)) + ...
                social_learning_factor * social_random * ...
                (global_best - particle_positions(:, particle_index));
        end
        
        % 限制速度
        for particle_index = 1:population_size
            velocity_limit_indices = 1;
            velocity_limit_factors = ones(dimension, 1);
            for dimension_index = 1:dimension
                if particle_velocities(dimension_index, particle_index) > velocity_max(dimension_index)
                    velocity_limit_factors(velocity_limit_indices) = ...
                        velocity_max(dimension_index) / particle_velocities(dimension_index, particle_index);
                    velocity_limit_indices = velocity_limit_indices + 1;
                elseif particle_velocities(dimension_index, particle_index) < velocity_min(dimension_index)
                    velocity_limit_factors(velocity_limit_indices) = ...
                        velocity_min(dimension_index) / particle_velocities(dimension_index, particle_index);
                    velocity_limit_indices = velocity_limit_indices + 1;
                end
            end
            velocity_limit_factor = min(velocity_limit_factors);
            for dimension_index = 1:dimension
                if particle_velocities(dimension_index, particle_index) > velocity_max(dimension_index)
                    particle_velocities(dimension_index, particle_index) = ...
                        particle_velocities(dimension_index, particle_index) * velocity_limit_factor;
                elseif particle_velocities(dimension_index, particle_index) < velocity_min(dimension_index)
                    particle_velocities(dimension_index, particle_index) = ...
                        particle_velocities(dimension_index, particle_index) * velocity_limit_factor;
                end
            end
        end
        
        % 更新位置
        particle_positions = particle_positions + particle_velocities;
        
        % 限制位置
        for particle_index = 1:population_size
            for dimension_index = 1:dimension
                if particle_positions(dimension_index, particle_index) > position_max(dimension_index)
                    particle_positions(dimension_index, particle_index) = position_max(dimension_index);
                elseif particle_positions(dimension_index, particle_index) < position_min(dimension_index)
                    particle_positions(dimension_index, particle_index) = position_min(dimension_index);
                end
            end
        end
        
        % 重新评估适应度，更新个体最优和全局最优
        for particle_index = 1:population_size
            particle_position = particle_positions(:, particle_index)';
            particle_new_fitness = fitness_function(particle_position, function_number);
            if particle_new_fitness < particle_fitness(particle_index)
                personal_bests(:, particle_index) = particle_positions(:, particle_index);
                particle_fitness(particle_index) = particle_new_fitness;
            end
            if particle_new_fitness < global_best_fitness
                global_best_fitness = particle_new_fitness;
            end
        end
        
        [global_best_fitness, global_best_index] = min(particle_fitness);
        global_best_fitness_history(iteration) = global_best_fitness;
        global_best = personal_bests(:, global_best_index);
    end
    
    % 绘制全局最优适应度变化曲线
    figure(1);
    plot(iteration_number, global_best_fitness_history, '-b');
    legend('优化后的 PSO');
    xlabel('迭代次数');
    ylabel('全局适应度');
    hold on;
end

function [particle_positions, particle_velocities] = initialize_particles(...
    dimension, population_size, position_max, position_min, ...
    velocity_max, velocity_min)
    % 初始化粒子的位置和速度
    for dimension_index = 1:dimension
        particle_positions(dimension_index, :) = ...
            position_min(dimension_index) + ...
            (position_max(dimension_index) - position_min(dimension_index)) * rand(1, population_size);
        particle_velocities(dimension_index, :) = ...
            velocity_min(dimension_index) + ...
            (velocity_max(dimension_index) - velocity_min(dimension_index)) * rand(1, population_size);
    end
end

function fitness = fitness_function(particle_position, function_number)
    % 计算粒子的适应度
    dimension = size(particle_position, 1);
    
    % 选择标准测试函数
    switch function_number
        case 1
            % f1_Sphere
            sphere_function = particle_position(:)' * particle_position(:);
            fitness = sphere_function;

        case 2
            % f2_Griewank
            griewank_term1 = particle_position(:)' * particle_position(:) / 4000;
            griewank_term2 = 1;
            for dimension_index = 1:dimension
                griewank_term2 = griewank_term2 * cos(particle_position(dimension_index) / sqrt(dimension_index));
            end
            griewank_function = griewank_term1 - griewank_term2 + 1;
            fitness = griewank_function;

        case 3
            % f3_Rastrigin
            rastrigin_function = particle_position(:)' * particle_position(:) - ...
                10 * sum(cos(particle_position(:) * 2 * pi)) + 10 * dimension;
            fitness = rastrigin_function;

        case 4
            % f4_Rosenbrock
            rosenbrock_term = 0;
            for dimension_index = 1:(dimension - 1)
                rosenbrock_term = rosenbrock_term + ...
                    100 * (particle_position(dimension_index + 1) - ...
                    particle_position(dimension_index)^2)^2 + ...
                    (particle_position(dimension_index) - 1)^2;
            end
            fitness = rosenbrock_term;
    end
end