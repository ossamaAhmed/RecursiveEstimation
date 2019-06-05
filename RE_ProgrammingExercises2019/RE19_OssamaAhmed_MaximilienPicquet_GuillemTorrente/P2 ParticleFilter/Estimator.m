function [postParticles] = Estimator(prevPostParticles, sens, act, estConst, km)
% The estimator function. The function will be called in two different
% modes: If km==0, the estimator is initialized. If km > 0, the
% estimator does an iteration for a single sample time interval using the 
% previous posterior particles passed to the estimator in
% prevPostParticles and the sensor measurement and control inputs.
%
% Inputs:
%   prevPostParticles   previous posterior particles at time step k-1
%                       The fields of the struct are [1xN_particles]-vector
%                       (N_particles is the number of particles) 
%                       corresponding to: 
%                       .x_r: x-locations of the robot [m]
%                       .y_r: y-locations of the robot [m]
%                       .phi: headings of the robot [rad]
%                           
%   sens                Sensor measurement z(k), scalar
%
%   act                 Control inputs u(k-1), [1x2]-vector
%                       act(1): u_f, forward control input
%                       act(2): u_phi, angular control input
%
%   estConst            estimator constants (as in EstimatorConst.m)
%
%   km                  time index, scalar
%                       corresponds to continous time t = k*Ts
%                       If tm==0 initialization, otherwise estimator
%                       iteration step.
%
% Outputs:
%   postParticles       Posterior particles at time step k
%                       The fields of the struct are [1xN_particles]-vector
%                       (N_particles is the number of particles) 
%                       corresponding to: 
%                       .x_r: x-locations of the robot [m]
%                       .y_r: y-locations of the robot [m]
%                       .phi: headings of the robot [rad]
%
%
% Class:
% Recursive Estimation
% Spring 2019
% Programming Exercise 2
%
% --
% ETH Zurich
% Institute for Dynamic Systems and Control
%

% Set number of particles:
N_particles = 1000; % obviously, you will need more particles than 10.

%% Mode 1: Initialization
if (km == 0)
    % Sample from the initial uniform distributions
    % Choose position in circle in polar coordinates
    r = estConst.d .* sqrt(rand(1, N_particles));
    theta = rand(1, N_particles) * 2 * pi;
    circ_x = r .* cos(theta);
    circ_y = r .* sin(theta);
    
    % Choose pA or pB
    circ_choice = randsample([1, 2], N_particles, true);
    circ_coords = [[estConst.pA]; [estConst.pB]];
    
    postParticles.x_r = transpose(circ_coords(circ_choice, 1)) + circ_x;
    postParticles.y_r = transpose(circ_coords(circ_choice, 2)) + circ_y;
    
    % Orientation
    postParticles.phi = (rand(1, N_particles)*2 - 1) * estConst.phi_0;
   
    return;
end

%% Mode 2: Estimator iteration.
% If km > 0, we perform a regular update of the estimator.

% Compute the polygonal shape 
room_polyshape = polyshape(estConst.contour);

% Prior Update:

% Compute process noises
v_f = (rand(1, N_particles)*2 - 1) * estConst.sigma_f;
v_phi = (rand(1, N_particles)*2 - 1) * estConst.sigma_phi;

% Apply system dynamics 
new_x_r = prevPostParticles.x_r + ...
    ((act(1) + v_f)) .* cos(prevPostParticles.phi);
new_y_r = prevPostParticles.y_r + ...
    ((act(1) + v_f)) .* sin(prevPostParticles.phi);
new_phi = prevPostParticles.phi + act(2) + v_phi;

% Check if particles are still within room
check = isinterior(room_polyshape, new_x_r, new_y_r);
[new_x_r, new_y_r, new_phi] = arrayfun(@select_update, check', ... 
    prevPostParticles.x_r, prevPostParticles.y_r, prevPostParticles.phi, ...
    new_x_r, new_y_r, new_phi);
postParticles.x_r = new_x_r;
postParticles.y_r = new_y_r;
postParticles.phi = new_phi;

% Posterior Update:

% Compute the distance of every particle to the facing wall
x_p = particle_measurement(postParticles, room_polyshape, estConst, N_particles);

% Calculate theoretical noise of every particle and probabilities
w = sens - x_p ;
p_w = noise_pdf(w, estConst.epsilon);
p_w = p_w / sum(p_w);

ratio_non_zeros = 1 - sum(p_w==0) / N_particles;
reg_w = -log(ratio_non_zeros + 0.05) + log(1.05);

% Scale each particle by the measurement likelihood
c_pdf = cumsum(p_w);
p_sample = rand(1, N_particles);

% Re-sample particles
k = bsxfun(@minus, p_sample(:)', c_pdf(:)); 
k(k > 0) = -inf;
[~, new_particles] = max(k);

sample_diversity_ratio = length(unique(new_particles))/N_particles;
reg_d = -log(sample_diversity_ratio + 0.1) + log(1.1);
reg_f = reg_w * reg_d;

if sample_diversity_ratio < 0.1
    new_particles = arrayfun(@randomize_particle_choice, new_particles, ...
        ones(1, N_particles) * N_particles);
end

postParticles.x_r = postParticles.x_r(new_particles);
postParticles.y_r = postParticles.y_r(new_particles);
postParticles.phi = postParticles.phi(new_particles);

% Roughening
K = 0.01;
d = 3;
E = [max(postParticles.x_r)-min(postParticles.x_r); ...
    max(postParticles.y_r)-min(postParticles.y_r); ...
    max(postParticles.phi)-min(postParticles.phi)];
s = K * E * N_particles ^ (-1/d) * reg_f;

postParticles.x_r = postParticles.x_r + normrnd(0, s(1), [1, N_particles]);
postParticles.y_r = postParticles.y_r + normrnd(0, s(2), [1, N_particles]);
postParticles.phi = postParticles.phi + normrnd(0, s(3), [1, N_particles]);

end % end estimator


function[particle] = randomize_particle_choice(particle, N)
    if rand > 0.9
        particle = unidrnd(N);
    end
end

function[selected_x_r, selected_y_r, selected_phi] = select_update(...
    check, prev_x_r, prev_y_r, prev_phi, new_x_r, new_y_r, new_phi)
if check
    selected_x_r = new_x_r;
    selected_y_r = new_y_r;
    selected_phi = new_phi;
else
    selected_x_r = prev_x_r;
    selected_y_r = prev_y_r;
    selected_phi = prev_phi;
end
end

function[p_w] = noise_pdf(noise_v, eps)
p_w = zeros(1, length(noise_v));
for i=1:length(noise_v)
    noise = noise_v(i);
    if noise < -3*eps 
        p_w(i) = 0;
    elseif noise < -2.5*eps
        p_w(i) = (noise + 3*eps) * 2/(5*eps^2);
    elseif noise < -2.0*eps
        p_w(i) = -(noise + 2*eps) * 2/(5*eps^2);
    elseif noise < 0
        p_w(i) = (noise + 2*eps) * 1/(5*eps^2);
    elseif noise < 2*eps
        p_w(i) = 2/(5*eps) - noise * 1/(5*eps^2);
    elseif noise < 2.5*eps
        p_w(i) = (noise - 2*eps) * 2/(5*eps^2);
    elseif noise < 3.0*eps
        p_w(i) = 1/(5*eps) - (noise - 2.5*eps)* 2/(5*eps^2);
    else
        p_w(i) = 0;
    end 
end
end

function[range_sensor] = particle_measurement(postParticles, ...
    room_polyshape, estConst, N)
    
    x_range = max(estConst.contour(:, 1)) - min(estConst.contour(:, 1));
    y_range = max(estConst.contour(:, 2)) - min(estConst.contour(:, 2));
    max_d = sqrt(x_range*2 + y_range*2)* 1.1;
    
    x_out = postParticles.x_r + cos(postParticles.phi) * max_d;
    y_out = postParticles.y_r + sin(postParticles.phi) * max_d;

    lineseg = zeros(2, 2, N);
    lineseg(1, 1, :) = postParticles.x_r;
    lineseg(1, 2, :) = postParticles.y_r;
    lineseg(2, 1, :) = x_out;
    lineseg(2, 2, :) = y_out;
    
    intersections = zeros(2, N);
    
    for i=1:N
        [~, out] = intersect(room_polyshape, lineseg(:, :, i));
        intersections(:, i) = out(1, :);
    end
    range_sensor = sqrt((postParticles.x_r - intersections(1, :)).^2 + ...
        (postParticles.y_r - intersections(2, :)).^2);
end