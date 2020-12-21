%% M4: PnP-ADMM (IRCNN)
% Requires MatConvNet to run, available at
% https://www.vlfeat.org/matconvnet/


% Choose your favoutite [320 320] denoising network
clear all
%rng(1234)

%% Choose model for final denoising

name2 = 'DnCNN3';
load DnCNN3
net2 = net;
net2 = vl_simplenn_tidy(net2);


%% Choose model for PnP-ADMM

name1 = 'IRCNN8';
sigmaModel = 8;
folderModel   = 'models';
load(fullfile(folderModel,'modelgray.mat'));
net = loadmodel(sigmaModel,CNNdenoiser);
net = vl_simplenn_tidy(net);

% Choose your test images
FilePath = 'testing_set\testingset_true';
% Set number of images to test
FileNum = 20;
delta = 1;

%% ADMM options
ADMMoptions.max_iter = 150;
ADMMoptions.delta =delta;
ADMMoptions.verbose = 1;
ADMMoptions.show_iters = 0; %Sampling rate to show iterations

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load Images
imdsTrue = imageDatastore(FilePath, 'IncludeSubfolders', true);
imdsTrue.ReadFcn = @(x) im2double(imread(x));

im = im2single(imread(imdsTrue.Files{1}));

disp('Started at');
disp(datetime);


disp('File start.');
%%Fourier Measurement
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


%% Load Image
x0 = im2double(imread(imdsTrue.Files{FileNum}));
%Compute the noise standard deviation according to the desired input SNR
isnr = 30;
sigma = norm(im)/sqrt(N)*10^(-isnr/20);
noise = sigma/sqrt(2)*(randn(N1,N2) + 1i*randn(N1,N2));
noise = reshape(ifftshift(mask.*fftshift(noise)),N,1);

y=Phit(x0);
y = y + noise;
bp_x = Phi(y );
bp_x = bp_x - min(bp_x(:));
bp_x = bp_x/max(bp_x(:));
SSIM = ssim(double(bp_x), x0);


figure;

tiledlayout(2,2);
nexttile, imagesc(real(bp_x)), colormap gray, axis image
title(['Backprojection, SSIM = ' num2str(SSIM)]);  
nexttile, imagesc(x0), axis image, colormap gray, title('Original'); 

%% Reconstruction for the noiseless case

%Estimation of the noise bound
epsilon1 = sigma*sqrt(M+2*sqrt(M));
tstart= tic;
xsol1 = PnP_ADMM(y,epsilon1,Phit,Phi,net, ADMMoptions);
tend1 = toc(tstart);
xsol2 = MatConvNetDenoise(xsol1, net2);

disp(['File end (' num2str(tend1) 'seconds).']);
%Compute SNR
rsnr = 20*log10(norm(x0(:))/norm(x0(:)-xsol1(:)));
SSIM = ssim(double(xsol1), x0);

nexttile, imagesc(real(xsol1)), axis image, colormap gray
title({['Reconstruction (N=' num2str(ADMMoptions.max_iter) ', ' num2str(round(tend1,1)) 's and delta = ' num2str(delta) '). RSNR=',num2str(rsnr),' dB, SSIM=' num2str(SSIM)],...
    ['PnP-ADMM using ' name1 ]});
rsnr = 20*log10(norm(x0(:))/norm(x0(:)-xsol2(:)));
SSIM = ssim(double(xsol2), x0);
nexttile, imagesc(real(xsol2)), axis image, colormap gray
title( {['RSNR=',num2str(rsnr),' dB, SSIM=' num2str(SSIM)],...
    ['Upscaled with ' name2]});

figure
imagesc(real(xsol2 - xsol1)), axis image, colormap gray
title('Upscaling difference')

figure
montage({bp_x, xsol1, x0}, 'Size', [1,3])
title('Backprojected Image (left), Recovered Image (middle) and True Image (right)');
DenX=MatConvNetDenoise(x0, net2);
figure
montage({bp_x, xsol2, x0, DenX}, 'Size', [2,2])
title('Clockwise from top left: Backprojected Image, Recovered Image + Denoise, Denoised Ground Truth, Ground Truth.');

function img = MatConvNetDenoise(nimg,net)
    nimg = single(nimg);
    Out = vl_simplenn(net,nimg);
    res = Out(end).x;
    img = double(nimg - res);
end


% function img = Denoise(nimg,net)
%     nimg = single(nimg);
%     Out = activations(net,nimg, 'Output');
%     img = double(Out);
% end

function xsol = PnP_ADMM(y,epsilon,Phit,Phi,net,options)
% Optional input arguments.
if ~isfield(options, 'max_iter'), options.max_iter = 35; end
if ~isfield(options, 'delta'), options.delta = 1; end


% Indicator Prox
sc = @(z) z*min(epsilon/norm(z(:)), 1); % scaling

%%Initializations.

%Dual variable.
v=zeros(size(y));

%Initial solution (all zero solution)
%Initial residual/intermediate variable
s =  - y;

%Initial l2 projection
n = sc(s) ;

%Creating the initial solution variable
%with all zeros
xsol = Phi(v);


for i = 1:options.max_iter

    xsol = denoise(xsol-options.delta*real(Phi(s+n-v)), net); 
    s = Phit(xsol) - y;
    n = sc(v-s);
    v = v -(s+n);
  
end


end

  function img = denoise(nimg,net)
    nimg = single(nimg);
    Out = vl_simplenn(net,nimg);
    res = Out(end).x;
    img = double(nimg - res);
end