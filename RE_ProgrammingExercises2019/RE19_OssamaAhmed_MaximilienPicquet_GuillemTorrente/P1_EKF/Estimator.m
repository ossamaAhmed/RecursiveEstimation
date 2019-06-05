function [posEst,linVelEst,oriEst,driftEst,...
          posVar,linVelVar,oriVar,driftVar,estState] = ...
    Estimator(estState,actuate,sense,tm,estConst)
% [posEst,linVelEst,oriEst,driftEst,...
%    posVar,linVelVar,oriVar,driftVar,estState] = 
% Estimator(estState,actuate,sense,tm,estConst)
%
% The estimator.
%
% The function is initialized for tm == 0, otherwise the estimator does an
% iteration step (compute estimates for the time step k).
%
% Inputs:
%   estState        previous estimator state (time step k-1)
%                   May be defined by the user (for example as a struct).
%   actuate         control input u(k-1), [1x2]-vector
%                   actuate(1): u_t, thrust command
%                   actuate(2): u_r, rudder command
%   sense           sensor measurements z(k), [1x5]-vector, INF entry if no
%                   measurement
%                   sense(1): z_a, distance measurement a
%                   sense(2): z_b, distance measurement b
%                   sense(3): z_c, distance measurement c
%                   sense(4): z_g, gyro measurement
%                   sense(5): z_n, compass measurement
%   tm              time, scalar
%                   If tm==0 initialization, otherwise estimator
%                   iteration step.
%   estConst        estimator constants (as in EstimatorConst.m)
%
% Outputs:
%   posEst          position estimate (time step k), [1x2]-vector
%                   posEst(1): p_x position estimate
%                   posEst(2): p_y position estimate
%   linVelEst       velocity estimate (time step k), [1x2]-vector
%                   linVelEst(1): s_x velocity estimate
%                   linVelEst(2): s_y velocity estimate
%   oriEst          orientation estimate (time step k), scalar
%   driftEst        estimate of the gyro drift b (time step k), scalar
%   posVar          variance of position estimate (time step k), [1x2]-vector
%                   posVar(1): x position variance
%                   posVar(2): y position variance
%   linVelVar       variance of velocity estimate (time step k), [1x2]-vector
%                   linVelVar(1): x velocity variance
%                   linVelVar(2): y velocity variance
%   oriVar          variance of orientation estimate (time step k), scalar
%   driftVar        variance of gyro drift estimate (time step k), scalar
%   estState        current estimator state (time step k)
%                   Will be input to this function at the next call.
%
%
% Class:
% Recursive Estimation
% Spring 2019
% Programming Exercise 1
%
% --
% ETH Zurich
% Institute for Dynamic Systems and Control
% Raffaello D'Andrea, Matthias Hofer, Carlo Sferrazza
% hofermat@ethz.ch
% csferrazza@ethz.ch
%

%% Initialization
if (tm == 0)
    % Do the initialization of your estimator here!
    
    p_start_bound = estConst.StartRadiusBound;
    orientation_start_bound = estConst.RotationStartBound; %uniform distribution
    gyro_drift_start_bound = estConst.GyroDriftStartBound;
    
    % initial state mean
    posEst = [0, 0]; % 1x2 matrix
    linVelEst = [0, 0]; % 1x2 matrix
    oriEst = [0]; % 1x1 matrix
    driftEst = [0]; % 1x1 matrix
    
    % initial state variance
    posVar = [(1/4)*p_start_bound^2, (1/4)*p_start_bound^2]; % 1x2 matrix
    linVelVar = [0, 0]; % 1x2 matrix
    oriVar = [(1/3)*orientation_start_bound]; % 1x1 matrix
    driftVar = [(1/3)*gyro_drift_start_bound]; % 1x1 matrix double check if they will change this value
    
    % estimator variance init (initial posterior variance)
    estState.Pm = diag([posVar,linVelVar,oriVar,driftVar]);
    % estimator state
    estState.xm = [posEst, linVelEst, oriEst, driftEst];
    % time of last update
    estState.tm = tm;
    return;
end

%% Estimator iteration.

% get time since last estimator update
dt = tm - estState.tm;
estState.tm = tm; % update measurement update time

% prior update
tspan = [tm-dt tm];

%getting the mean
x_hat_prev = estState.xm;
P_m_prev = estState.Pm;
n_states = 6;
x_hat_P_t_prev = [x_hat_prev'; reshape(P_m_prev, [n_states * n_states, 1])];

[~, x_hat_P_t] = ode45(@(t, x_hat_P_t) priorUpdate(x_hat_P_t, actuate, estConst), tspan, x_hat_P_t_prev);
x_hat_P_t = x_hat_P_t(end,:)';

x_hat_prior = x_hat_P_t(1:n_states);
P_prior = reshape(x_hat_P_t(n_states + 1:end),[n_states n_states]);

% measurement update

isThirdStationAvailable = isfinite(sense(3));

if (isThirdStationAvailable)
    [H, M, R, h] = get_measurments_matrices(x_hat_prior, estConst);
else
    [H, M, R, h] = get_measurments_matrices_missing_reliable(x_hat_prior, estConst);
    sense = [sense(1:2), sense(4:5)];
end
%update kalman gain
K = (P_prior * H') * inv((H * P_prior * H')+ (M * R * M'));
%update mean posterior
x_hat_m = x_hat_prior + K*(sense' - h);
%update variance posterior
P_m = (eye(6) - (K * H)) * P_prior;

% Set resulting estimates and variances
estState.xm = x_hat_m';
estState.Pm = P_m;
% Output quantities
posEst = estState.xm(1:2);
linVelEst = estState.xm(3:4);
oriEst = estState.xm(5);
driftEst = estState.xm(6);

posVar = [estState.Pm(1, 1), estState.Pm(2, 2)];
linVelVar = [estState.Pm(3, 3), estState.Pm(4, 4)];
oriVar = [estState.Pm(5, 5)];
driftVar = [estState.Pm(6, 6)];


function [x_hat_P_t_derivative] = priorUpdate(x_hat_P_t, u, estConst)
    n_states = 6;
    x_hat_ = x_hat_P_t(1:n_states);
    P_t = reshape(x_hat_P_t(n_states + 1:end), [n_states, n_states]);
    
    P_t_derivative_ = approximate_P_t(x_hat_, P_t, estConst, u);
    
    x_hat_derivative = q(x_hat_, estConst, u);
    x_hat_P_t_derivative = [x_hat_derivative; P_t_derivative_];
end

function  [P_t_derivative] = approximate_P_t(x_hat, P_t, estConst, u)
    A_matrix = get_A_matrix(x_hat, estConst, u);
    L_matrix = get_L_matrix(x_hat, u, estConst);
    Q_c = diag([estConst.DragNoise, estConst.RudderNoise, estConst.GyroDriftNoise]);
    P_t_derivative = (A_matrix * P_t) + (P_t * A_matrix') + (L_matrix * Q_c * L_matrix');
    
    P_t_derivative = reshape(P_t_derivative, [numel(P_t_derivative), 1]);
end



%dynamics equations to get the mean
%state = [pos, linvel, orientation, drift]
%input = [u_t, u_r]
function [x_hat_derivative] = q(x_hat, estConst, u)
    C_d = estConst.dragCoefficient;
    C_r = estConst.rudderCoefficient;
    
    x_hat_derivative = [
        x_hat(3); 
        x_hat(4);
        cos(x_hat(5)) * ( tanh(u(1)) - (C_d * (x_hat(3)^2 + x_hat(4)^2)) ); %missing the process noise V_d
        sin(x_hat(5)) * ( tanh(u(1)) - (C_d * (x_hat(3)^2 + x_hat(4)^2)) ); %missing the process noise v_d
        C_r * u(2); %missing the process noise V_r
        0; %missing the process noise V_b
    ];
end

%get A matrix
function [A_matrix] = get_A_matrix(x_hat, estConst, u)
    C_d = estConst.dragCoefficient;
    A_matrix = zeros(6);
    A_matrix(1, 3) = 1;
    
    A_matrix(2, 4) = 1;
    
    A_matrix(3, 3) = -2 * C_d * cos(x_hat(5)) * x_hat(3);
    A_matrix(3, 4) = -2 * C_d * cos(x_hat(5)) * x_hat(4);
    A_matrix(3, 5) = -sin(x_hat(5)) * (tanh(u(1)) - C_d * (x_hat(3)^2 + x_hat(2)^2));
    
    A_matrix(4, 3) = -2 * C_d * sin(x_hat(5)) * x_hat(3);
    A_matrix(4, 4) = -2 * C_d * sin(x_hat(5)) * x_hat(4);
    A_matrix(4, 5) = cos(x_hat(5)) * (tanh(u(1)) - C_d * (x_hat(3)^2 + x_hat(2)^2));
end

%get L matrix
function [L_matrix] = get_L_matrix(x_hat, u, estConst)
    C_d = estConst.dragCoefficient;
    C_r = estConst.rudderCoefficient;
    
    L_matrix = zeros(6, 3);
    
    L_matrix(3, 1) = -cos(x_hat(5)) * C_d * (x_hat(3)^2 + x_hat(4)^2);
    
    L_matrix(4, 1) = -sin(x_hat(5)) * C_d * (x_hat(3)^2 + x_hat(4)^2);
    
    L_matrix(5, 2) = C_r * u(2);
    
    L_matrix(6, 3) = 1;
end

%mesaurment model
%z = [a(x, y), b (x, y), c (x, y), g, n]
function [H, M, R, h] = get_measurments_matrices(x, estConst)
    z = zeros(5, 1);
    z(1) = sqrt((x(1) - estConst.pos_radioA(1))^2 + (x(2) - estConst.pos_radioA(2))^2); % without measurment noise
    z(2) = sqrt((x(1) - estConst.pos_radioB(1))^2 + (x(2) - estConst.pos_radioB(2))^2); % without measurment noise
    z(3) = sqrt((x(1) - estConst.pos_radioC(1))^2 + (x(2) - estConst.pos_radioC(2))^2); % without measurment noise
    z(4) = x(5) + x(6);
    z(5) = x(5);
    
    H = zeros(5, 6);
    H(1, 1) = (x(1) - estConst.pos_radioA(1))/z(1);
    H(1, 2) = (x(2) - estConst.pos_radioA(2))/z(1);
    
    H(2, 1) = (x(1) - estConst.pos_radioB(1))/z(2);
    H(2, 2) = (x(2) - estConst.pos_radioB(2))/z(2);
    
    H(3, 1) = (x(1) - estConst.pos_radioC(1))/z(3);
    H(3, 2) = (x(2) - estConst.pos_radioC(2))/z(3);
    
    H(4, 5) = 1;
    H(4, 6) = 1;
    
    H(5, 5) = 1;
    
    M = eye(5);
    
    R = diag([estConst.DistNoiseA, estConst.DistNoiseB, estConst.DistNoiseC, estConst.GyroNoise, estConst.CompassNoise]);
    
    h = z;

end

%mesaurment model
%z = [a(x, y), b (x, y), c (x, y), g, n]
function [H, M, R, h] = get_measurments_matrices_missing_reliable(x, estConst)
    z = zeros(4, 1);
    z(1) = sqrt((x(1) - estConst.pos_radioA(1))^2 + (x(2) - estConst.pos_radioA(2))^2); % without measurment noise
    z(2) = sqrt((x(1) - estConst.pos_radioB(1))^2 + (x(2) - estConst.pos_radioB(2))^2); % without measurment noise
    z(3) = x(5) + x(6);
    z(4) = x(5);
    
    H = zeros(4, 6);
    H(1, 1) = (x(1) - estConst.pos_radioA(1))/z(1);
    H(1, 2) = (x(2) - estConst.pos_radioA(2))/z(1);
    
    H(2, 1) = (x(1) - estConst.pos_radioB(1))/z(2);
    H(2, 2) = (x(2) - estConst.pos_radioB(2))/z(2);
    
    
    H(3, 5) = 1;
    H(3, 6) = 1;
    
    H(4, 5) = 1;
    
    M = eye(4);
    
    R = diag([estConst.DistNoiseA, estConst.DistNoiseB, estConst.GyroNoise, estConst.CompassNoise]);
    
    h = z;

end


end
