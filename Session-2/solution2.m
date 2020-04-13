%%  Solution sesion 2

close All
clear All



%% Step 1: Load the images

% a) Load the first image
imFirst=imread('faces/s1/1.pgm');
figure;
imshow(imFirst)

% b) Load the training images (40 persons X 7 images)

nTrainImages=40*7; % number of training images
[height, width] =size(imFirst); % The images have the same size. Use the first one to caclulate the number of pixels 

imagesTrain=zeros( nTrainImages, height*width ); % Matrix with the training images as rows 

% c) read training images.
for i=1:40
    for j=1:7  %seven images for training.
        tmp_img=imread( sprintf('faces/s%d/%d.pgm',i,j) );
        
        index = (i-1)*7 + j; 
        imagesTrain(index,:)=tmp_img(:); %this copy images as rows.
    end
end

% Transpose it to go to have images as columns
imagesTrain = imagesTrain';
size(imagesTrain)



% d) read testing images.
nTestImages=40*3; % testing images
imagesTest=zeros(nTestImages, height*width);

for i=1:40
    for j=1:3  %3 images for testing.
        tmp_img=imread( sprintf('faces/s%d/%d.pgm',i,j + 7) );
        
        index = (i-1)*3 + j; 
        imagesTest(index,:)=tmp_img(:);
    end
end

imagesTest = imagesTest';
size(imagesTest)


%% Step 2: PCA

% a) find the mean image
mean_face = mean(imagesTrain, 2); % by default mean caclulates the mean column wise

figure 
img_mean = reshape(mean_face, height, width); 
imagesc( img_mean), colormap gray, axis off


%  b) mean-shifted input images
shifted_images = imagesTrain - repmat(mean_face,1, nTrainImages ); %repmat replicates matrix


% This is a pre-step to optimize calculation. (see your lecture notes)
% Compute Y' Y 
YY = 1/size(shifted_images, 1) * (shifted_images' * shifted_images);

% c) Compute eigenvectors
[evectors, evalues] = eig(YY);
evalues = diag(evalues);  %get only eigenvalues  (eig, returns a diagonal matrix)

% Solve Y = evector
evectors = shifted_images* evectors;

% d) Sort eigenvectors/eigenvalues
[~, isorted] = sort(-1*evalues);  %multiplied to -1 to sort in decreasing order..
evalues = evalues(isorted); %using the indexes of sorted data assing new evalues
evectors = evectors(:, isorted); %using the indexes of sorted data assing new evectors


% e) only retain the top 'num_eigenfaces' eigenvectors (i.e. the principal components)
num_eigenfaces = 30;
evectors = evectors(:, 1:num_eigenfaces);

% Normalize the eigenvectors so || evector_i| | = 1
for i = 1: num_eigenfaces
    evectors(:,i) = evectors(:,i)/norm(evectors(:,i));
end


% f) project the images into the subspace to generate the feature vectors
features = evectors' * shifted_images;


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

% b) display the eigenvalues
normalised_evalues = evalues / sum(evalues);
figure, plot(cumsum(normalised_evalues));
xlabel('No. of eigenvectors'), ylabel('Variance accounted for');
xlim([1 100]), ylim([0 1]), grid on;


%% Step 4: Identify a person

testIm=9; %i chosed person 9 
input_image=imagesTest(:,testIm);

% calculate the similarity of the input to each training image
feature_vec = evectors' * (input_image(:) - mean_face);

similarity_score = arrayfun(@(n) 1 / (1 + norm(features(:,n) - feature_vec)), 1:nTrainImages);

% find the image with the highest similarity
[match_score, match_ix] = max(similarity_score);

% display the result
figure, imshow([uint8(reshape(imagesTest(:,testIm), height, width)) uint8(reshape(imagesTrain(:,match_ix),  height, width))]);

title(sprintf('matches %d, score %f', match_ix, match_score));

%% Step 5: Cluster the persons

% a) Cluster the original images
clusters = kmeans(imagesTrain', 40);


% Show the the images that were assigned to cluster 1
img_index = find(clusters == 40);

for i = 1 : min( length(img_index), 7)
    figure(111);

    subplot(1, 7, i)
    img = reshape(imagesTrain(:, img_index(i)), height, width);
    imagesc(img), colormap gray, axis off
end



% b) Now cluster on the reduced space
clusters_pca = kmeans(features', 40);

% Show the the images that were assigned to cluster 1
img_index = find(clusters_pca == 12);
for i = 1 : min( length(img_index), 7)
    figure(111);
    subplot(1, 7, i)

    img = reshape(imagesTrain(:, img_index(i)), height, width);
    imagesc(img), colormap gray, axis off
end



%%
% c) Try agglomerative clustering on some images
checkImages = 8:21;
Z_agglo = linkage(features(:,checkImages)', 'single');



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



