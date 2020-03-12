close all;
clear;

%% Step 1: Load the images

nImages = 40*10;
nTrainImages = 40*7;

% --- LOAD THE FIRST IMAGE
imFirst=imread('faces/s1/1.pgm');
figure;
imshow(imFirst);
% The images have the same size. Use the first one to calculate the number
% of pixels.
[height, width] = size(imFirst); 

% --- LOAD ALL TRAIN IMAGES
images = zeros(width*height, 40*10);
counter = 1;
for dir = 1:40
    for file = 1:7
        img = imread(['faces/s', num2str(dir), '/', num2str(file), '.pgm']);
        imshow(img);
        images(:,counter) = img(:);
        counter = counter + 1;
    end 
end

% --- LOAD ALL TEST IMAGES
for dir = 1:40
    for file = 8:10
        img = imread(['faces/s', num2str(dir), '/', num2str(file), '.pgm']);
        imshow(img);
        images(:,counter) = img(:);
        counter = counter + 1;
    end 
end

% --- TRAIN / TEST IMAGES
imagesTrain = images(:, 1:nTrainImages);
imagesTest = images(:, nTrainImages+1:end);


%% Step 2: PCA

% --- FIND THE MEAN IMAGE
figure;
mean_face = mean(imagesTrain, 2);
img_mean = reshape(mean_face, size(imFirst));
imagesc(img_mean), colormap gray, axis off;
title('Mean Face');


% --- SUBTRACT THE MEAN
shifted_images = imagesTrain - repmat(mean_face, 1, nTrainImages);


% This is a pre-step to optimize calculation. (see your lecture notes)
%%%TODO: normally we should find the covariance of shifted_images but it will be a 10304*10304 matrix!
%%%to be optimized and avoid "out of memory" error, we use a trick! (see
%%%chapter 6 of the book (p. 164)
%%%Compute Y'*Y 
YY = 1/size(shifted_images, 1) * (shifted_images' * shifted_images);

% c) Compute eigenvectors
[evectors, evalues] = eig(YY);
evalues = diag(evalues);
evectors = shifted_images * evectors;

% d) Sort eigenvectors based on their corresponding eigenvalues
[evalues, idx] = sort(evalues, 'descend');
evectors = evectors(:, idx);

% e) only retain the top 'num_eigenfaces' eigenvectors (i.e. the principal components)
num_eigenfaces = 30;
evectors = evectors(:, 1:num_eigenfaces);

% Normalize the eigenvectors so || evector_i|| = 1
for i = 1: num_eigenfaces
    evectors(:,i) = evectors(:,i)/norm(evectors(:,i));
end


% f) project the images into the subspace to generate the feature vectors
coords = evectors' * shifted_images;

% g) reconstruct images
images_rec = evectors * coords;
images_rec = images_rec + repmat(mean_face, 1, nTrainImages);

% Compare:
figure;
subplot(1,2,1);
imagesc(reshape(images(:,50), size(imFirst)));
colormap gray, axis off;
title('Original Image');
subplot(1,2,2);
imagesc(reshape(images_rec(:,50), size(imFirst)));
colormap gray, axis off;
title(['Reconstructed with ', num2str(num_eigenfaces), ' eigs']);


%% Step 3: Analysis of eigen faces
% a) display the eigenvectors
figure;
for n = 1:num_eigenfaces
    subplot(3, ceil(num_eigenfaces/3), n);
    evector = reshape(evectors(:,n),  height, width);
    imagesc(evector)
    colormap('gray');
    axis off
end

% b) display the commulative eigenvalues
figure;
plot(cumsum(evalues)/max(cumsum(evalues)));
title('Cumsum of eigenvalues (%)');
%%%TODO: hint: "cumsum" function for accumulation


%% Step 4: Identify a person
testIm=9; %i chosed person 9 
input_image=imagesTest(:,testIm);

% calculate the similarity of the input to each training image
% PDF: s(y1,y2) = (1+||y1-y2||)^-1
%      0 = infinitely far apart, 1 = same
shifted_input = input_image - mean_face;
coords_input = evectors' * shifted_input;
similarity_score = 1./(1+vecnorm(coords-coords_input));
%%%TODO: put the result in "similarity_score" variable

% find the image with the highest similarity
[match_score, match_ix] = max(similarity_score);

% display the result
figure, imshow([uint8(reshape(imagesTest(:,testIm), height, width)) uint8(reshape(imagesTrain(:,match_ix),  height, width))]);

title(sprintf('matches to the %d th training sample with score %f', match_ix, match_score));

%% Step 5: Cluster the persons

% a) Cluster the original images
[clusters, centroids] = kmeans(imagesTrain', 40);
centroids = centroids';
%%%TODO: put the results in "clusters" variable. "clusters" indicates cluster number for each sample. hint: use kmeans function
%%

% Show the the images that were assigned to cluster 1
img_index = find(clusters == 1);

for i = 1 : min( length(img_index), 7)
    figure(111);
    subplot(1, 7, i)
    img = reshape(imagesTrain(:, img_index(i)), height, width);
    imagesc(img), colormap gray, axis off
end

%%

% b) Now cluster on the reduced space
%%%TODO: put the results in "clusters_pca" variable.

% Show the the images that were assigned to cluster 1
img_index = find(clusters_pca == 1);
for i = 1 : min( length(img_index), 7)
    figure(111);
    subplot(1, 7, i)

    img = reshape(imagesTrain(:, img_index(i)), height, width);
    imagesc(img), colormap gray, axis off
end

% c) Try agglomerative clustering on some images
%%%TODO: hint: use linkage command for agglomerative clustering with
%%%"single" metric and put the results in "Z_agglo" variable

figure;
[H,T] = dendrogram(Z_agglo,'colorthreshold','default');
c = cluster(Z_agglo,'maxclust',2);


% Show the the images that were assigned to cluster 1
img_index = find(c == 1);
for i = 1 : min( length(img_index), 5)
    figure(111);
    subplot(1, 5, i)

    img = reshape(imagesTrain(:, checkImages(img_index(i)) ), height, width);
    imagesc(img), colormap gray, axis off
end



