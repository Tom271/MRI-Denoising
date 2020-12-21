%% M1.m file
%
% %
% This file is for your implementation and validation of M1 (sections 3 and 4).
% It also contains useful functions (sections 1 and 2). 
testset_filepath = 'testing_set/testingset_true/';
addpath ../Lab2_students/Lab2_students/Lab2/ADMM/
% Fixed the seed for random generator
rng(243);

%% Load an image
im=imread('file1000073_15.png');
%normalization
im=im2double(im);
im=im/max(max(im));

%% 1. Fourier measurements and noise operators

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
disp(w);
num_meas = num_meas-w;

% Building the mask
mask = zeros(N1,N2);
lines_int = randi(N1,[num_meas,1]); % Sampling uniformly at random
mask(floor(N1/2-w):floor(N2/2+w),:) = 1;
mask(lines_int,:) = 1;
mask(1,:) = 0;
mask(N1,:) = 0;
mask = mask.';

% % Definition of the measurement operators Phit and Phi
Phit = @(x) reshape(ifftshift(mask.*fftshift(fft2(x))),N,1)/sqrt(N);
Phi = @(x) real(ifft2(ifftshift(mask.*fftshift(reshape(x,N1,N2)))))*sqrt(N);

%Signal measurements
y0 = Phit(im);

%Compute the noise standard deviation according to the desired input SNR
isnr = 30;
sigma = norm(im)/sqrt(N)*10^(-isnr/20);
noise = sigma/sqrt(2)*(randn(N1,N2) + 1i*randn(N1,N2));
noise = reshape(ifftshift(mask.*fftshift(noise)),N,1);

y = y0 + noise;

outName = "y_TEST.png";
mim = mask.*fftshift(fft2(im));
imwrite(mat2gray(log10(real(abs(mim))+1e-3)), outName);

HtY = real(Phi(y));

epsilon = sigma*sqrt(M + 2*sqrt(M));



%% 2. Sparsity operators

% Selection of the wavelet mode
xd = Phi(y);
nlevel=8;
wv='db4';
dwtmode('per');
[alphad,S]=wavedec2(xd,nlevel,wv);

% Definition of the sparsity operators Psi and Psit
Psit = @(x) wavedec2(x, nlevel,wv); 
Psi = @(x) waverec2(x,S,wv);
%% 3. M1 implementation ...
% Use the functions provided above as appropriate
% Get list of images
imagefiles = dir([testset_filepath, '*.png']);

for ii=1:length(imagefiles)
   currentfilename = imagefiles(ii).name;
   currentimage = imread(currentfilename);
   currentimage=im2double(currentimage);
   currentimage=currentimage/max(max(currentimage));
   images{ii} = currentimage;
end

%Options for the admm solver
options.verbose = 2;
options.rel_tol = 1e-6;
options.rel_tol2 = 1e-6;
options.max_iter = 5000;
options.delta = 1;

options.rho = 3.7276;

indices = [14 19 20];   %choose some images
img_cell=cell(3,3);


for i=1:length(indices)
    j =indices(i);
    % Build new mask
    mask = zeros(N1,N2);
    lines_int = randi(N1,[num_meas,1]); % Sampling uniformly at random
    mask(floor(N1/2-w):floor(N2/2+w),:) = 1;
    mask(lines_int,:) = 1;
    mask(1,:) = 0;
    mask(N1,:) = 0;
    mask = mask.';

    % Define new measurement operators Phit and Phi
    Phit = @(x) reshape(ifftshift(mask.*fftshift(fft2(x))),N,1)/sqrt(N);
    Phi = @(x) real(ifft2(ifftshift(mask.*fftshift(reshape(x,N1,N2)))))*sqrt(N);

    %Signal measurements
    im = cell2mat(images(j));
    sigma = norm(im)/sqrt(N)*10^(-isnr/20);
    noise = sigma/sqrt(2)*(randn(N1,N2) + 1i*randn(N1,N2));
    noise = reshape(ifftshift(mask.*fftshift(noise)),N,1);
    y = Phit(im) + noise;
    epsilon = sigma*sqrt(M + 2*sqrt(M));       

    xd = Phi(y);
    [alphad,S]=wavedec2(xd,nlevel,wv);
    xd = real(xd); %backprojected image

    % Definition of the sparsity operators Psi and Psit
    Psit = @(x) wavedec2(x, nlevel,wv); 
    Psi = @(x) waverec2(x,S,wv);

    % Run ADMM
    tstart=tic;
    [xsol1, fval1, niter1] = admm_conbpdn(y,epsilon,Phit,Phi,Psi,Psit,options);
    tend1=toc(tstart);
    disp("t= " +tend1);
    times(i) = tend1;

    img_cell(1,i)={xd};
    img_cell(2,i)={real(xsol1)};
    img_cell(3,i)={im};

    %Compute SNR
    snr1 = 20*log10(norm(im(:))/norm(im(:)-xsol1(:)));
    SNR(j) = snr1;
    %Compute SSIM
    SSIM(j) = ssim(xsol1,im);

    disp("Image " + j + " done!");
end

%% 4. M1 validation ...
 montage({img_cell{1,1},img_cell{1,2},img_cell{1,3}...

         img_cell{2,1},img_cell{2,2},img_cell{2,3}...

         img_cell{3,1},img_cell{3,2},img_cell{3,3}},'Size',[3 3]);