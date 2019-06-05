function [postParticles] = Estimator(prevPostParticles, sens, act, estConst, km)
% The estimator function. The function will be called in two different
% modes: If km==1, the estimator is initialized. If km > 0, the
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
N_particles = 500; % obviously, you will need more particles than 10.
wkm1 = repmat(1/N_particles, N_particles, 1);

% boundaries 
bounds = polyshape(estConst.contour);

%% Mode 1: Initialization
if (km == 0)
    % Do the initialization of your estimator here!
    
    
    % generate particles
    for i = 1:N_particles 
        prevPostParticles.x_r(1,i) = datasample([ ...
            unifrnd(estConst.pA(1)-estConst.d, estConst.pA(1)+estConst.d), ...
            unifrnd(estConst.pB(1)-estConst.d, estConst.pB(1)+estConst.d)], ...
            1);
        prevPostParticles.y_r(1,i) = datasample([ ...
            unifrnd(estConst.pA(2)-estConst.d, estConst.pA(2)+estConst.d), ...
            unifrnd(estConst.pB(2)-estConst.d, estConst.pB(2)+estConst.d)], ...
            1);
        prevPostParticles.phi(1,i) = unifrnd(-estConst.phi_0,estConst.phi_0);        
    end
    
    postParticles.x_r = prevPostParticles.x_r; % 1xN_particles matrix
    postParticles.y_r = prevPostParticles.y_r; % 1xN_particles matrix
    postParticles.phi = prevPostParticles.phi; % 1xN_particles matrix
        
    % and leave the function
    return;
end % end init

%% Mode 2: Estimator iteration.
% If km > 0, we perform a regular update of the estimator.

% Implement your estimator here!

% memory 
wk = zeros(N_particles, 1);
z_est = zeros(N_particles,1);
particleStates = zeros(N_particles,3);

x = prevPostParticles.x_r;
y = prevPostParticles.y_r;
phi = prevPostParticles.phi;

% Prior Update:
for i = 1:N_particles
    %sampling
    particleStates(i,1) = x(i)+(act(1)+unifrnd(-estConst.sigma_f,estConst.sigma_f))*cos(phi(i));
    particleStates(i,2) = y(i)+(act(1)+unifrnd(-estConst.sigma_f,estConst.sigma_f))*sin(phi(i));
    particleStates(i,3) = phi(i)+act(2)+unifrnd(-estConst.sigma_phi,estConst.sigma_phi); 

    % est measurements
    lineseg = [x(i) y(i); x(i)+3*cos(phi(i)) y(i)+3*sin(phi(i))];
    [~,out] = intersect(bounds, lineseg);
    x_C = out(1,1);
    y_C = out(1,2);
    
    wki = 0; % measurement noise
    z_est(i) = sqrt((x(i)-x_C)^2+(y(i)-y_C)^2) + wki;
    prior = sens - z_est(i);
    
    % weights
    sigma_v = sqrt(0.001);
    wk(i) = wkm1(i) * normpdf(prior, 0, sigma_v);    
end


% Posterior Update:
wk = wk./sum(wk);

% effective sapling size
Neff = 1/sum(wk.^2);

% Resampling 
resample_perc = 0.85;
Nt = resample_perc * N_particles;
if Neff < Nt
    idx = randsample(1:N_particles, N_particles, true, wk);
    particleStates = particleStates(idx, :);
    
    wk = repmat(1/N_particles, 1, N_particles);
    
end 

particleFinalState = zeros(1,3); 
for i = 1:N_particles
    particleFinalState = particleFinalState + wk(i)*particleStates(i,:);
end


postParticles.x_r = particleStates(:,1);
postParticles.y_r = particleStates(:,2);
postParticles.phi = particleStates(:,3);

end % end estimator