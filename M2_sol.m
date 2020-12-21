% load trained network results
load('adam_30_5e-3_piecewise_8_shuffled_train95val5.mat')
% M2\TrainedNets\
% file structure etc.
cd testing_set/testingset_true/     % go to training_true folder
file_list = dir('*.png');               % image list 
L = length(file_list);                  % number of images

% allocate space for arrays and cells
rsnr_test = zeros(L,2);
ssim_test = zeros(L,2);
tend = zeros(L,1);
img_cell = cell(L,3);

% loop through each image in testingset_true
for l = 1:L
    im = imread(file_list(l).name);      % read image
    im = im2double(im);
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
%     disp(w);
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
    
    % Compute the noise standard deviation according to the desired input SNR
    isnr = 30;
    sigma = norm(im)/sqrt(N)*10^(-isnr/20);
    noise = sigma/sqrt(2)*(randn(N1,N2) + 1i*randn(N1,N2));
    noise = reshape(ifftshift(mask.*fftshift(noise)),N,1);
    
    % Signal measurements
    y0 = Phit(im);
    
    % Add noise
    y = y0 + noise;
    %% 2. Backprojection
    bp = real(Phi(y));
    %% 3. Normalise to [0,1]
    bp = bp - min(bp(:));
    bp = bp/max(bp(:));
    %% 4. Feed to pretrained network
    tstart = tic;                               % start timer
    m2_img = double(predict(M2_net,bp));    % predicted image
    tend(l) = toc(tstart);                      % stop timer
    
    % calculate rsnr and ssim
    rsnr_test(l,1) = 20*log10(norm(im(:))/norm(im(:)-bp(:)));           % rsnr of backprojected image
    rsnr_test(l,2) = 20*log10(norm(im(:))/norm(im(:)-m2_img(:)));       % rsnr of predicted image
    ssim_test(l,1) = ssim(im,bp);                                       % ssim of backprojected image
    ssim_test(l,2) = ssim(im,m2_img);                                   % ssim of predicted image
    
    % save images
    img_cell{l,1} = bp;         % backprojected image
    img_cell{l,2} = m2_img;     % predicted image
    img_cell{l,3} = im;         % true image
    %% 5. Write backprojected image
    if 0
        var = file_list(l).name;
        outName = "bp_"+var+".png";
        cd ..\testingset_dirty\
        imwrite(bp, outName);

        cd ..\..\testing_set\testingset_true\      % go back to training_true folder
    end
end
%% display results
disp('----- BP results ------')
disp(['mean rsnr of backprojected images is ', num2str(mean(rsnr_test(:,1)))])
disp(['std rsnr of backprojected images is ', num2str(std(rsnr_test(:,1)))])
disp(['mean ssim of backprojected images is ', num2str(mean(ssim_test(:,1)))])
disp(['std ssim of backprojected images is ', num2str(std(ssim_test(:,1)))])

% show std rsnr and ssim
disp('----- Predicted results ------')
disp(['mean rsnr of predicted images is ', num2str(mean(rsnr_test(:,2)))])
disp(['std rsnr of predicted images is ', num2str(std(rsnr_test(:,2)))])
disp(['mean ssim of predicted images is ', num2str(mean(ssim_test(:,2)))])
disp(['std ssim of predicted images is ', num2str(std(ssim_test(:,2)))])
disp(['mean time of predicted images is ', num2str(mean(tend))])
disp(['std time of predicted images is ', num2str(std(tend))])
%% show a sample set of images
n = randi(L-2);    % random integer between 1 and L-2
figure;
montage({img_cell{n,1},img_cell{n,2},img_cell{n,3},...
         img_cell{n+1,1},img_cell{n+1,2},img_cell{n+1,3},...
         img_cell{n+2,1},img_cell{n+2,2},img_cell{n+2,3}},'Size',[3 3]);   % put noisy, true and predicted images together
title('Backprojected Image (left), Predicted Image (middle) and True Image (right)')