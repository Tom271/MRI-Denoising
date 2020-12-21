patchSize = 64;
depth = 17;
channels = 1;

% % Given an image im, we create the mask
% Useful variables
ft = 4;             % Subsampling rate
p = 0.08;           % Width (in percent) of the central band)
N1 = 320;
N2 = 320;
N = N1*N2;
num_meas = floor(N1/ft);
M = num_meas*N2;    % Total number of measurements
w = floor(N1*p/2); 
disp(w);
num_meas = num_meas-w;


load("fulltrain_dncnn_sigma-17_11_15__16_23.mat", "trainedNet")
layers_transfer = trainedNet.Layers(2:end); % Select all the layers except the first one.
layers = [imageInputLayer([320,320], 'Name','Input', 'Normalization', 'none')
     layers_transfer]; % Create the layers of the new network
lgraph = layerGraph(layers); % Build the graph
mynet = assembleNetwork(lgraph); % Assemble the graph as a network

trainDataPath = 'training_set\training_set\trainingset_true';

cd 'testing_set\testingset_true'      % go to training_true folder
file_list = dir('*.png');               % image list 
L = length(file_list); 


my_snrs = zeros(L,1);
my_ssims = zeros(L,1);
my_times = zeros(L,1);


for i = 1:L
    file = file_list(i).name
    test_im = imread(file);
    test_im = im2single(test_im);




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
    y = Phit(test_im);
    
    sigma = norm(test_im(:))/sqrt(N) * 10^(-isnr/20);
    noise = (randn(size(y)) + 1i*randn(size(y)))*sigma/sqrt(2);
    noise = reshape(ifftshift(reshape(mask,N,1).*fftshift(noise)), N,1);

    y = y + noise;
    

    options.max_iter = 100;
    options.delta = 1;
    options.reltol = 3.0;
    epsilon = sigma * sqrt(M + 2 * sqrt(M));
    tic;
    xsol = pnp_admm(y, epsilon, Phit, Phi, mynet,test_im, options);
    my_times(i) = toc;
    figure
    imshow(xsol)
    
    my_snrs(i) = snr(test_im, xsol - test_im);
    
    my_ssims(i) = ssim(test_im, single(xsol));

end

