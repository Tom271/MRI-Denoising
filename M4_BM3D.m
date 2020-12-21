%% M4: ADMM-BM3D
% This script requires bm3d.m (v3.0.7) to run, available from
% https://www.cs.tut.fi/~foi/GCF-BM3D/ 

rng(2020)
train_date = '18-Dec-2020'; % Date trained

options.max_iter = 50;
options.delta = 1;
options.mu = 1;
options.DT = 0.1;
options.rho = 70;

%Parameters for the TV prox
paramtv.max_iter = 1000;
paramtv.rel_obj = 1e-3;
paramtv.verbose = 0;


% data:
snrs = zeros(1, 20);
ssims = zeros(1, 20);
times = zeros(1, 20);
%% Read test data
test_set = imageDatastore('testing_set/testingset_true', 'IncludeSubFolders', true, ...
    'ReadFcn', @NormalizeImageResize);
%%  Fourier measurements and noise operators

% Loop through images
for i=1:20
    disp(i)
    im = read(test_set);  % Read image i
    % % Given an image im, we create the mask
    % Useful variables
    ft = 4;             % Subsampling rate
    p = 0.08;           % Width (in percent) of the central band)
    N1 = size(im,1);
    N2 = size(im,2);
    N = N1*N2;
    num_meas = floor(N1/ft);
    M = num_meas*N2;    % Total number of measurements
    w = floor(N1*p/2); 
    num_meas = num_meas-w;

    % Building the mask
    mask = zeros(N1,N2);
    lines_int = randi(N1,[num_meas,1]); % Sampling uniformly at random
    mask(floor(N1/2-w):floor(N2/2+w),:) = 1;
    mask(lines_int,:) = 1;
    mask(1,:) = 0;
    mask(N1,:) = 0;
    mask = mask.';

    % Definition of the measurement operators Phit and Phi
    Phit = @(x) reshape(ifftshift(mask.*fftshift(fft2(x))),N,1)/sqrt(N);
    Phi = @(x) real(ifft2(ifftshift(mask.*fftshift(reshape(x,N1,N2)))))*sqrt(N);

    % Noise addition operator
    isnr = 30;
    sigma = norm(im)/sqrt(N)*10^(-isnr/20);
    noise = sigma/sqrt(2)*(randn(N1,N2) + 1i*randn(N1,N2));
    noise = reshape(ifftshift(mask.*fftshift(noise)),N,1);
    add_noise = @(y) y + (randn(size(y)) + 1i*randn(size(y)))*sigma/sqrt(2);
    
    % Observe noisy fourier measurements
    y = Phit(im) + noise;
    epsilon = sigma*sqrt(M + 2*sqrt(M)); % Noise factor
    
    tic;
    [imstar, t] = admm_bm3d(y, epsilon, Phit, Phi, options);
    times(i) = toc;
    disp(times(i));
    snrs(i) = 20*log10(norm(im(:), 2)/norm(im(:) - imstar(:), 2));
    ssims(i) = ssim(im2double(im2double(imstar)), im2double(im));
    figure(i)
    imshow(imstar)
%     title('Backprojected Image (left), Predicted Image (middle) and True Image (right)', 'FontSize', 20)
    saveas(gcf, ['M4_Figures/full_reconstruction', num2str(i),'final.png'])
end

stats.mean_snr = mean(snrs);    stats.mean_ssim = mean(ssims);
stats.sd_snr = std(snrs);       stats.sd_ssim = std(ssims);
stats.mean_time = mean(times);  stats.sd_time = std(times);

% Save data
save(strcat('results', train_date))
%% Functions

% Normalization to the training and testing datasets
% Working with a normalized data set can help enhance the training of
% networks. This can be achieved with the following function.
function img_res = NormalizeImageResize(file)
%   This function takes as input:
%       - file: an image directory.
%   It returns:
%       - img_res: the image in range [0,1] with dimensions [256x256].
    img = imread(file);
    img_res = im2double(img);

end

function [xsol, t] = admm_bm3d(y, epsilon, Phit, Phi, options)
% ADMM based algorithm to solve the following problem:
%
%   min ||Psit x||_1   s.t.  ||y - Phit x||_2 < epsilon,
%
% which is posed as the equivalent problem:
%
%   min ||Psit x||_1   
%
%   s.t.  n = y - Phit x and ||n||_2 < epsilon.
%
% Inputs:
%
% - y is the measurement vector. 
% - epsilon is a bound on the residual error.
% - Phit is the forward measurement operator and Phi the associated adjoint 
%   operator.
% - Psit is a sparfying transform and Psi its adjoint.
% - options is a Matlab structure containing the following fields:
% 
%   - verbose: 0 no log, 1 print main steps, 2 print all steps.
%
%   - rel_tol: minimum relative change of the objective value (default:
%     1e-4). The algorithm stops if
%
%           | f( x(t) ) - f( x(t-1) ) | | / | f( x(t) ) | < rel_tol
%     and   ||y - Phit x(t)||_2 < epsilon,
%
%     where x(t) is the estimate of the solution at iteration t.
%
%   - rel_tol2: second condition for stopping the algorithm. rel_tol2 is 
%     the minimum relative change of the iterates (default:
%     1e-4). The algorithm stops if
%           || x(t) - x(t-1) ||_2 | / || x(t) ||_2 < rel_tol2
%
%     where x(t) is the estimate of the solution at iteration t.
%
%   - max_iter: max. number of iterations (default: 200).
%
%   - rho: penalty parameter for ADMM (default: 1e2).
%
%   - delta: step size for the proximal gradient update (default: 1e0).
%
%   To guarantee convergence, the following condition is needed:
%
%      delta*L <= 1
%
%      where L is the square norm of the operator Phit, i.e. the square of
%      its maximum singular value.
% 
% Outputs:
%
% - xsol: solution of the problem.
%
% - fval : objective value.
%
% - t : number of iterations used by the algorithm.

% Optional input arguments.
if ~isfield(options, 'verbose'), options.verbose = 1; end
if ~isfield(options, 'rel_tol'), options.rel_tol = 1e-4; end
if ~isfield(options, 'rel_tol2'), options.rel_tol2 = 1e-4; end
if ~isfield(options, 'max_iter'), options.max_iter = 200; end
if ~isfield(options, 'rho'), options.rho = 1e2; end
if ~isfield(options, 'delta'), options.delta = 1; end

%Useful functions.
sc = @(z) z*min(epsilon/norm(z(:)), 1); % scaling
%Dual variable.
v=zeros(size(y));

%Initial solution (all zero solution)
%Initial residual/intermediate variable
s =  - y;

%Initial l2 projection
n = sc(s) ;

%Creating the initial solution variable
%with all zeros
xsol = im2double(zeros(size(Phi(s))));
t = 0;
%% Main loop. 
while  t <= options.max_iter
    xsol = BM3D(xsol - options.delta*real(Phi(s + n - v)), 0.07);
    s = Phit(xsol) - y;
    n = sc(v - s);
    v = v - (s + n);
    t = t + 1;
end
end

