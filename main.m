clc;clear;
    addpath('meanshift');
    addpath('LBP');
    addpath('Tamura');
    addpath('libsvm/');
    addpath('./libsvm/matlab/');
    addpath('./etract_feature/');
    addpath('./utils/');
    addpath('./LIME-master/');
    addpath('./LBP/');
    url ='E:\shadow removal\AISD\Train412\shadow\chicago33_sub16.tif';   
    im = imread(url);
 disp 'Segmenting'
    [dummy seg] = edison_wrapper(im, @RGB2Luv, ...
       'SpatialBandWidth',10, 'RangeBandWidth',5, ...
       'MinimumRegionArea',200); 
    seg = seg + 1;
    segnum = max(max(seg));  
    marker=zeros(size(seg));
    [m,n]=size(seg);
    for i=1:m
        for j=1:n
            top=seg(max(1,i-1),j);
            bottom=seg(min(m,i+1),j);
            left=seg(i,max(1,j-1));
            right=seg(i,min(n,j+1));
            if ~(top==bottom && bottom==left && left==right)
                marker(i,j)=1;
            end
        end
    end
    figure,imshow(marker);
    I3=im;
    for i=1:m
        for j=1:n
            if marker(i,j)==1
                I3(i,j,:)=0;
            end
        end
    end
figure,imshow(I3);


disp 'Detect'
gray1=imread('E:\shadow removal\AISD\Train412\mask\chicago33_sub16.tif');
gray1=uint8(255*gray1);
BW3=gray1;
tic
ind={};
    for iReg=1:segnum
        ind{iReg} = seg(:)==iReg;
    end 
shapemean = calcShapeMean(seg, segnum);
area = shapemean.area;
centroids = shapemean.center;

label = zeros([1, segnum]) + 255;
n_nonshadow = segnum;
flag = 0;
for i = 1:segnum       
        if area(i)~=0&&mode(BW3((seg==i)))==255
            label(i) = 0;
            n_nonshadow = n_nonshadow - 1;
            flag = flag + 1;
        end
end
shadow=label(seg);
%%%%%%%%%%%%%%%%%%%%%%zengqiang%%%%%%%%%%%
[Ti, Tout, img_out, Iout] = lime_main_module(im,0.01, 1.188,10, 1.5, 1);
%figure,imshow(img_out)
% img_out=im;
im_result=uint8(zeros(size(im,1),size(im,2),3));
for i=1:size(im,1)
    for j=1:size(im,2)
        for k=1:size(im,3)
            if shadow(i,j)==0
                im_result(i,j,k)=255*img_out(i,j,k);
            else
                im_result(i,j,k)=im(i,j,k);
            end
        end
    end
end
img_out=im_result;
% %figure,imshow(im_result)
im=im_result;
% img_out=im;
%%%%%%%%%%%%阴影区域分割%%%%%%%%%
for i=1:size(im,1)
    for j=1:size(im,2)
        if shadow(i,j)==0
           im_shadow(i,j,1)=im(i,j,1);
            im_shadow(i,j,2)=im(i,j,2);
            im_shadow(i,j,3)=im(i,j,3); 
        else 
            im_shadow(i,j,1)=0;
            im_shadow(i,j,2)=0;
            im_shadow(i,j,3)=0;
        end
    end
end
im_shadow=uint8(im_shadow);
   [dummy_shadow seg_shadow] = edison_wrapper(im_shadow , @RGB2Luv, ...
       'SpatialBandWidth',10, 'RangeBandWidth',5, ...
       'MinimumRegionArea',50); 
     seg_shadow = seg_shadow + 1;
    segnum_shadow = max(max(seg_shadow));  
    marker_shadow=zeros(size(seg_shadow));
%%%%%%%%取中心坐标%%%%%%%%%    
 ind={};
    for iReg=1:segnum_shadow
        ind{iReg} = seg_shadow(:)==iReg;
    end 
shapemean = calcShapeMean(seg_shadow, segnum_shadow);
area_shadow = shapemean.area;
centroids_shadow = shapemean.center;   

label_shadow = zeros([1, segnum_shadow]) + 255;
n_nonshadow = segnum_shadow;
flag = 0;
for i = 1:segnum_shadow       
        if area_shadow(i)~=0&&mode(shadow(seg_shadow==i))==0%%mode(BW3((seg==i)))
            label_shadow(i) = 0;
            n_nonshadow = n_nonshadow - 1;
            flag = flag + 1;
        end
end
%figure,imshow(label_shadow(seg_shadow))
[m,n]=size(seg_shadow);
for i=1:m
    for j=1:n
        top=seg_shadow(max(1,i-1),j);
        bottom=seg_shadow(min(m,i+1),j);
        left=seg_shadow(i,max(1,j-1));
        right=seg_shadow(i,min(n,j+1));
        if ~(top==bottom && bottom==left && left==right)
            marker_shadow(i,j)=1;
        end
    end
end
%figure,imshow(marker_shadow);
I4=im_shadow;
for i=1:m
    for j=1:n
        if marker_shadow(i,j)==1
            I4(i,j,:)=0;
        end
    end
end
%figure,imshow(I4);
for i=1:size(im,1)
    for j=1:size(im,2)
        if shadow(i,j)==0
          im_sunlit(i,j,1)=0;  
          im_sunlit(i,j,2)=0; 
          im_sunlit(i,j,3)=0; 
        else 
            im_sunlit(i,j,1)=im(i,j,1);
            im_sunlit(i,j,2)=im(i,j,2);
            im_sunlit(i,j,3)=im(i,j,3);
        end
    end
end
   [dummy_sunlit seg_sunlit] = edison_wrapper(im_sunlit , @RGB2Luv, ...
       'SpatialBandWidth',10, 'RangeBandWidth',15, ...
       'MinimumRegionArea',200); 
     seg_sunlit = seg_sunlit + 1;
    segnum_sunlit = max(max(seg_sunlit));  
    marker_sunlit=zeros(size(seg_sunlit));
[m,n]=size(seg_sunlit);
for i=1:m
    for j=1:n
        top=seg_sunlit(max(1,i-1),j);
        bottom=seg_sunlit(min(m,i+1),j);
        left=seg_sunlit(i,max(1,j-1));
        right=seg_sunlit(i,min(n,j+1));
        if ~(top==bottom && bottom==left && left==right)
            marker_sunlit(i,j)=1;
        end
    end
end
figure,imshow(marker_sunlit);
I5=im_sunlit;
for i=1:m
    for j=1:n
        if marker_sunlit(i,j)==1
            I5(i,j,:)=0;
        end
    end
end
%figure,imshow(I5);
%%%%%%%%取中心坐标%%%%%%%%%    
 ind={};
    for iReg=1:segnum_sunlit
        ind{iReg} = seg_sunlit(:)==iReg;
    end 
shapemean = calcShapeMean(seg_sunlit, segnum_sunlit);
area_sunlit = shapemean.area;
centroids_sunlit = shapemean.center;   

label_sunlit = zeros([1, segnum_sunlit]) + 255;
n_nonshadow = segnum_sunlit;
flag = 0;
for i = 1:segnum_sunlit       
        if area_sunlit(i)~=0&&mode(shadow(seg_sunlit==i))==0%%mode(BW3((seg==i)))
            label_sunlit(i) = 0;
            n_nonshadow = n_nonshadow - 1;
            flag = flag + 1;
        end
end
figure,imshow(label_sunlit(seg_sunlit))

for i=1:m
    for j=1:n
         marker_result(i,j)=marker_shadow(i,j)+marker_sunlit(i,j);       
    end
end
%figure,imshow(marker_result)
toc
%%%%%%%%%%最近邻匹配%%%%%%%%%%%%%%%%%
tic
x_shadow = pingjun(seg_shadow, segnum_shadow);
x_sunlit = pingjun(seg_sunlit, segnum_sunlit);
m=1;
for i = 1:segnum_shadow
    if label_shadow(i) == 0
        C(m,1:2)=[x_shadow(i,1),x_shadow(i,2)];
            m=m+1;
    end 
end
n=1;
for i = 1:segnum_sunlit
    if label_sunlit(i) == 255
        D(n,1:2)=[x_sunlit(i,1),x_sunlit(i,2)];
        n=n+1;
    end
end
[value,id]=find(label_shadow==0);
[value1,id1]=find(label_sunlit==255);
between = zeros([size(D, 1),size(C, 1)]);
epsilon=0.01;
for i = 1:size(D, 1)
    for j = 1:size(C, 1)
       distance = [D(i,1), D(i,2);C(j,1), C(j,2)];
       distance = sqrt((distance(1,1)-distance(2,1))^2 + (distance(1,2)-distance(2,2))^2);
       between(i, j)=distance;
    end
end
%num=round(0.8*size(D,1));
num=30;
near_x=zeros([size(C,1),num]);
for i = 1:size(C, 1)
    [b,N]=sort(between(:,i));
    near_x(i,1:num)=N(1:num);
end
%%%%%%第一列是阴影区域，第2-6列是光照区域 
 k=1;
 for i=1:size(C,1)
     near_xx(i,1:num+1)=[id(k),id1(near_x(i,1:num))];
     k=k+1;
 end
toc
%%%%%%%%%%%%%%%%%%特征匹配%%%%%%%%%%%%%%%%%%
tic
distance1=zeros(segnum_sunlit,segnum_shadow);
im_hsi = rgb2hsv(im); 
radius=1;
n=8;

im_gray=rgb2gray(img_out);

%figure,imshow(im_gray) %--zhx
im_gray=double(im_gray)/255;
%figure,imshow(im_gray) %--zhx





[gx gy] = gradient(im_gray);
gx(1:416,1) = 0;
gx(1:416,416) = 0;
gy(1,1:416) = 0;
gy(416,1:416) = 0;


gx=LBP(gx,radius,n);
gy=LBP(gy,radius,n);
%%[gxx gyy] = gradient(double(im_gray),2);
gx=double(gx)/255;
gy=double(gy)/255;

im_rgb=img_out;
im_r=im_rgb(:,:,1);

n=8
im_r=LBP(im_r,radius,n);
im_r=double(im_r)/255;




im_g=im_rgb(:,:,2);
im_g=LBP(im_g,radius,n);
im_g=double(im_g)/255;


im_b=im_rgb(:,:,3);
im_b=LBP(im_b,radius,n);
im_b=double(im_b)/255;


for i=2:size(im,1)-1
    for j=2:size(im,2)-1
%         gx1(i-2,j-2)=gx(i,j);
%         gy1(i-2,j-2)=gy(i,j);
%         im_r1(i-2,j-2)=im_r(i,j);
%         im_g1(i-2,j-2)=im_g(i,j);
%         im_b1(i-2,j-2)=im_b(i,j);
         seg_shadow1(i-1,j-1)=seg_shadow(i,j);
        seg_sunlit1(i-1,j-1)=seg_sunlit(i,j);
    end
end






%%%%%阴影区域计算奇异值矩阵结果、
U=zeros(5,5);
S=zeros(5,5);
V=zeros(5,5);
U1=zeros(5,5);
S1=zeros(5,5);
V1=zeros(5,5);



%%%%%%%%%%改进前写法
% % for k=1:segnum_shadow
% %     if label_shadow(k)==0
% %         u1= mean(im_r(seg_shadow1(:)==k));
% %         u2=mean(im_g(seg_shadow1(:)==k));
% %         u3=mean(im_b(seg_shadow1(:)==k));
% %         u4=mean(gx(seg_shadow1(:)==k));
% %         u5=mean(gy(seg_shadow1(:)==k));
% %         z0=zeros(5,5);
% %         z_R=zeros(5,5);
% %         for i = 1:size(im_r,1)
% %             for j=1:size(im_r,2)
% %                  if seg_shadow1(i,j)==k
% %                     z=[
% %                         im_r(i,j)-u1,im_g(i,j)-u2,im_b(i,j)-u3,gx(i,j)-u4,gy(i,j)-u5];                   
% %                     z_R=z'*z;
% %                     z0=z0+z_R;%%-zhx
% %                  end
% %                  %%z0=z0+z_R;%%
% %             end
% %         end  
% %         number=sum(seg_shadow1(:)==k);
% %         zz_R=z0./number;
% %         [U,S,V] = svd(zz_R);
% %     end
% %     %%%%%非阴影区域计算奇异值结果
% %     for m=1:segnum_sunlit
% %         if label_sunlit(m)==255
% %             z0=zeros(5,5);
% %             z_R=zeros(5,5);
% %             u1= mean(im_r(seg_sunlit1(:)==m));
% %             u2=mean(im_g(seg_sunlit1(:)==m));
% %             u3=mean(im_b(seg_sunlit1(:)==m));
% %             u4=mean(gx(seg_sunlit1(:)==m));
% %             u5=mean(gy(seg_sunlit1(:)==m));
% %             for i = 1:size(im_r,1)
% %                 for j=1:size(im_r,2)
% %                      if seg_sunlit1(i,j)==m
% %                         z=[im_r(i,j)-u1,im_g(i,j)-u2,im_b(i,j)-u3,gx(i,j)-u4,gy(i,j)-u5];                  
% %                         z_R=z'*z;
% %                         z0=z0+z_R;%%--zhx
% %                      end
% %                      %%z0=z0+z_R;%%
% %                 end
% %             end  
% %             number=sum(seg_sunlit1(:)==m);
% %             zz_R=z0./number;
% %             [U1,S1,V1] = svd(zz_R);
% %         end    
% %         for i=1:size(S,1)
% %             distance1(m,k)=distance1(m,k)+sqrt((S(i,i)-S1(i,i))^2);
% %         end
% %     end
% %     a=k;
% % end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% %%%%%%%%%%改进后写法
%阴影
for k=1:segnum_shadow
    if label_shadow(k)==0
        u1= mean(im_r(seg_shadow1(:)==k));
        u2=mean(im_g(seg_shadow1(:)==k));
        u3=mean(im_b(seg_shadow1(:)==k));
        u4=mean(gx(seg_shadow1(:)==k));
        u5=mean(gy(seg_shadow1(:)==k));
        z0=zeros(5,5);
        z_R=zeros(5,5);
        for i = 1:size(im_r,1)
            for j=1:size(im_r,2)
                 if seg_shadow1(i,j)==k
                    z=[
                        im_r(i,j)-u1,im_g(i,j)-u2,im_b(i,j)-u3,gx(i,j)-u4,gy(i,j)-u5];                   
                    z_R=z'*z;
                    z0=z0+z_R;%%-zhx
                 end
                 %%z0=z0+z_R;%%
            end
        end  
        number=sum(seg_shadow1(:)==k);
        zz_R=z0./number;
        [U,S,V] = svd(zz_R);
    end
    shadow_Features{k} = S;
end
%%%%%非阴影区域计算奇异值结果
for m=1:segnum_sunlit
    if label_sunlit(m)==255
        z0=zeros(5,5);
        z_R=zeros(5,5);
        u1= mean(im_r(seg_sunlit1(:)==m));
        u2=mean(im_g(seg_sunlit1(:)==m));
        u3=mean(im_b(seg_sunlit1(:)==m));
        u4=mean(gx(seg_sunlit1(:)==m));
        u5=mean(gy(seg_sunlit1(:)==m));
        for i = 1:size(im_r,1)
            for j=1:size(im_r,2)
                 if seg_sunlit1(i,j)==m
                    z=[im_r(i,j)-u1,im_g(i,j)-u2,im_b(i,j)-u3,gx(i,j)-u4,gy(i,j)-u5];                  
                    z_R=z'*z;
                    z0=z0+z_R;%%--zhx
                 end
                 %%z0=z0+z_R;%%
            end
        end  
        number=sum(seg_sunlit1(:)==m);
        zz_R=z0./number;
        [U1,S1,V1] = svd(zz_R);
    end 
    sunlit_Features{m} = S1;
end

for k=1:segnum_shadow
    for m=1:segnum_sunlit
        for i=1:5
            distance1(m,k)=distance1(m,k)+sqrt((shadow_Features{k}(i,i)-sunlit_Features{m}(i,i))^2);
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
distance2=distance1;
for i=1:size(distance1,2)
    if label_shadow(i)==255
            distance2(:,i)=1000000000000;
    end
end
for j=1:size(distance1,1)
    if label_sunlit(j)==0
        distance2(j,:)=100000000000;
    end
end
% % 测试直接按照特征距离排序
% % % % % for j = 1:size(distance2, 2) 
% % % % %      near_2(j,1)=j;
% % % % %     %for i=1:size(distance2, 1)
% % % % %     [b,N]=sort(distance2(2:size(distance2,1),j));     %%---zhx 这里是不是应该只排2-6行 才是根据距离筛选过的光照块 否则相当于完全按照特征距离排列了
% % % % %         %%%%[b,N]=sort(near_1(2:6,i));  ---zhx
% % % % %      near_2(j,2:6)=N(1:5);    
% % % % %     %end
% % % % % end
% % % % % for i = 1:size(near_2, 2) 
% % % % %     near_2(i,2:6)=near_2(i,2:6)+1;
% % % % % end
%38阴影
for i=1:size(near_xx,1)
 near(1,i)=near_xx(i,1);
end
 
for i=1:size(near_xx,1)
    for j=2:size(near_xx,2)
       %near_3(1,j-1)=near_2(i,1);
       near(j,i)=distance2(near_xx(i,j),near_xx(i,1)); %%此处根据距离筛选了 可能的特征值
    end
end
near_1=near;
near=near';



for i = 1:size(near_1, 2) 
    near_2(i,1)=near_1(1,i);
    [b,N]=sort(near_1(2:size(near_1,1),i));     
    %%%%[b,N]=sort(near_1(2:6,i));  ---zhx
    near_2(i,2:6)=N(1:5);    
end

for i = 1:size(near_1, 2) 
near_2(i,2:6)=near_2(i,2:6)+1;
end

for i = 1:size(near_xx, 1) 
    for j=2:size(near_2,2)
   near_2(i,j)=near_xx(i,near_2(i,j));
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%查看每块阴影对应的光照块
im1=uint8(im(:,:,1));
im2=uint8(im(:,:,2));
im3=uint8(im(:,:,3));
imR=uint8(zeros(size(im1,1),size(im1,2)));
imG=uint8(zeros(size(im2,1),size(im2,2)));
imB=uint8(zeros(size(im3,1),size(im3,2)));
imRef1=imR;
imRef2=imG; 
imRef3=imB;
 ret_result=uint8(zeros(size(im1,1),size(im1,2),3));   
 for k = 1:size(near_2, 1)
      for i=1:size(im,1)
          for j=1:size(im,2) 
              if seg_sunlit (i,j)==near_2(k,2)
                  imRef1(i,j)=im1(i,j);
                  imRef2(i,j)=im2(i,j);
                  imRef3(i,j)=im3(i,j);
              end
          end
      end
      %%ret_Ref=cat(3,imRef1,imRef2,imRef3); 
      %%%%%%ret_Ref = im_sunlit; ---zhx
      %figure,imshow(ret_Ref) 

 end 
ret_Ref=cat(3,imRef1,imRef2,imRef3); 
toc
%%%%%%ret_Ref = im_sunlit; ---zhx
figure,imshow(ret_Ref) 
imwrite(ret_Ref,'ret_Ref.tif')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%初始去除%%%%%%%%%%%%%%%%%%%%%%%
tic
%读取图像
source =im_shadow;

target =ret_Ref;  %参考图像  一般没有0

%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:size(shadow,1)
    for j =1:size(shadow,2)
        if shadow(i,j)==0
            shadow1(i,j)=255;
        else
            shadow1(i,j)=0;
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
I1=shadow1;  %待匀色图像的 黑白二值图  用来确定黑色区域位置
src=im_sunlit;  %要和结果图像相加
%转换数据类型
im2doubles=im2double(source);    %把图像转成0-1 
im2doublet=im2double(target);
% im2double转为双精度 并缩放为[0-1]
% reshape 是变为3列  带个 ‘ 是转置
%都是3行  列数是像素数  
rgb_s = reshape(im2doubles,[],3)';      
rgb_t = reshape(im2doublet,[],3)';
%A(find(A==0))=[];%找到A中0的位置，并令其为空，即删除
%去0   得是RGB三个通道 也即是对应矩阵同一列的都为0 才去0
%根据二值图的非黑色像素位置 先计算非0像素数目n 就是矩阵的列数 创建矩阵  一个个填充  
[a,b] = size(I1);   % a行b列对应矩阵 矩阵是按列排的  所以黑色像素（i，j）对应矩阵每行的  a*(j-1)+i
n=0;
for j=1:b  %第几列
    for i =1:a  %第几行
            if (I1(i,j)~=0)
              n=n+1;
            end
    end
end
disp(n);
%创建一个3行n列的矩阵  通过遍历图像对每一行矩阵 赋值 
m=n/3;
m1=1;m2=1;m3=1;
rgb_s1=zeros(3,n);  %生成m×n全零阵。
for j=1:b  %第几列
    for i =1:a  %第几行
            if (I1(i,j)~=0)
            rgb_s1(1,m1)= rgb_s(1,a*(j-1)+i);
            if(m1<n)
                m1=m1+1;
            end
            rgb_s1(2,m2)= rgb_s(2,a*(j-1)+i);
            if(m2<n)
                m2=m2+1;
            end
            rgb_s1(3,m3)= rgb_s(3,a*(j-1)+i);
            if(m3<n)
                m3=m3+1;
            end
            
            end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[a,b,k] = size(target);   % a行b列对应矩阵 矩阵是按列排的  所以黑色像素（i，j）对应矩阵每行的  a*(j-1)+i
n=0;
for j=1:b  %第几列
    for i =1:a  %第几行
            if (target(i,j,1)~=0)
              n=n+1;
            end
    end
end
disp(n);
%创建一个3行n列的矩阵  通过遍历图像对每一行矩阵 赋值 
m=n/3;
m1=1;m2=1;m3=1;
rgb_t1=zeros(3,n);  %生成m×n全零阵。
for j=1:b  %第几列
    for i =1:a  %第几行
            if (target(i,j,1)~=0)
            rgb_t1(1,m1)= rgb_t(1,a*(j-1)+i);
            if(m1<n)
                m1=m1+1;
            end
            rgb_t1(2,m2)= rgb_t(2,a*(j-1)+i);
            if(m2<n)
                m2=m2+1;
            end
            rgb_t1(3,m3)= rgb_t(3,a*(j-1)+i);
            if(m3<n)
                m3=m3+1;
            end
            
            end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%5.5%%%%%%%%%%
%计算均值
mean_s = mean(rgb_s1,2);         %  mean（A,2）计算每一行均值 结果为列向量
mean_t = mean(rgb_t1,2);
 %协方差矩阵
cov_s = cov(rgb_s1');    %协方差矩阵
cov_t = cov(rgb_t1');
%奇异值分解
[U_s,A_s,~] = svd(cov_s);  % 奇异值分解 
[U_t,A_t,~] = svd(cov_t);
rgbh_s = [rgb_s1;ones(1,size(rgb_s1,2))];
% translations 平移矩阵
T_t = eye(4); T_t(1:3,4) =  mean_t; %Target Image
T_s = eye(4); T_s(1:3,4) = -mean_s; %Source Image
% rotations   旋转矩阵  矩阵每一行末尾都有省略号，省略号是用于一行没输完，在第二行上接着输入的需要，
R_t = blkdiag(U_t,1);
R_s = blkdiag(inv(U_s),1);
% scalings :   缩放矩阵
% Note : for S_t ,after searching about previous work , we found that the
% S_t values must be taken as sqrt of the eigenvalues.(see original paper)
S_t = diag([diag(A_t).^(0.5);1]);
S_s = diag([diag(A_s).^(-0.5);1]);
%操作
rgbh_e = T_t*R_t*S_t*S_s*R_s*T_s*rgbh_s;   % estimated RGBs
rgb_e = rgbh_e(1:3,:);
rgb_e1=im2uint8(rgb_e);
% rgb_e1是三维矩阵  每一行分别代表R G B 
%先创建一个  全0的和 原图像大小相同的图像
%zeros(size(A))：生成与矩阵A相同大小的全零阵。
%先把target 变为灰度图 找非0的位置
R_e1=zeros(size(I1));   %结果的R通道
G_e1=zeros(size(I1));   %结果的G通道
B_e1=zeros(size(I1));   %结果的B通道
%对R通道 填充  把rgb_e1的第一行按顺序填充到R中   其中graytarget 一定是按列遍历
col=size(rgb_e1,2);
%[a,b] = size(I1); 
n1=1;
for j=1:b
    for i =1:a
            if (I1(i,j) ~=0)
                R_e1(i,j)=rgb_e1(1,n1);
                if(n1<col)
                n1=n1+1;
                end
            end
    end
end
disp(n1);
n2=1;
for j=1:b
    for i =1:a
            if (I1(i,j) ~=0)
                G_e1(i,j)=rgb_e1(2,n2);
                if(n2<col)
                n2=n2+1;
                end
            end
    end
end

n3=1;
for j=1:b
    for i =1:a
            if (I1(i,j) ~=0)
                B_e1(i,j)=rgb_e1(3,n3);
                if(n3<col)
                n3=n3+1;
                end
            end
    end
end

IR1=cat(3,R_e1,G_e1,B_e1);

%IR1=reshape(IR1,size(target));
IR1=uint8(IR1);
res=uint8(IR1)+src;
toc
figure; 
subplot(3,3,2);imshow(source); title('待匀色'); axis off
subplot(3,3,1);imshow(target); title('参考图像'); axis off
subplot(3,3,3);imshow(uint8(IR1));title('Result Image '); axis off
subplot(3,3,4); imshow(uint8(res));title('Result '); axis off
figure,imshow(res)
imwrite(res,'baseline_colortransfer.tif')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%精细去除%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
im = imread(url);
R=im(:,:,1);
G=im(:,:,2);
B=im(:,:,3);
RR=0;
RG=0;
RB=0;
imr=res(:,:,1);
img=res(:,:,2);
imb=res(:,:,3);
UR=0;
UG=0;
UB=0;
UR=average(IR1(:,:,1));
UG=average(IR1(:,:,2));
UB=average(IR1(:,:,3));
% UR=average(im_sunlit(:,:,1));
%  UG=average(im_sunlit(:,:,2));
%  UB=average(im_sunlit(:,:,3));
 im5=im_shadow;
for m=1:size(near_2,1)
UR=0;
UG=0;
UB=0;
RR=0;
RG=0;
RB=0;
    UR=average(imr(seg_shadow==near_2(m,1)));
    UG=average(img(seg_shadow==near_2(m,1)));
    UB=average(imb(seg_shadow==near_2(m,1)));
    SR=average(R(seg_shadow==near_2(m,1)));
    SG=average(G(seg_shadow==near_2(m,1)));
    SB=average(B(seg_shadow==near_2(m,1)));

        RR=(UR-SR)/SR;
        RG=(UG-SG)/SG;
        RB=(UB-SB)/SB;
%     RR_removal=RR/(size(near_2,2)-1);
%     RG_removal=RG/(size(near_2,2)-1);
%     RB_removal=RB/(size(near_2,2)-1);
    for i=1:size(im,1)
        for j=1:size(im,2)
            if seg_shadow(i,j)==near_2(m,1)
                im5(i,j,1)=(RR+1)*im(i,j,1);
                im5(i,j,2)=(RG+1)*im(i,j,2);
                im5(i,j,3)=(RB+1)*im(i,j,3);
            end
        end
    end
%     figure,imshow(im5);
end  
figure,imshow(im5);
sunlit=uint8(zeros(size(shadow,1),size(shadow,2),3));
for i=1:size(shadow,1)
    for j=1:size(shadow,2)
        for k=1:3
        if shadow(i,j)==255
            sunlit(i,j,1)=im(i,j,1);
            sunlit(i,j,2)=im(i,j,2);
            sunlit(i,j,3)=im(i,j,3);
        else
             sunlit(i,j,1)=0;
             sunlit(i,j,2)=0;
             sunlit(i,j,3)=0;
        end
        end
    end
end
removal1=im5+sunlit  ;
figure,imshow(removal1,[]) 
imwrite(removal1,'finalresult.tif')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%multi-scale sharppen%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
src=IR1;
figure;imshow(src), title('ԭͼ');
Radius=5;
%[ dest ] = multiScaleSharpen( src, Radius);
w1=0.3;
w2=0.45;
w3=0.95;
%%
sigma1 = 1.0;
sigma2 = 2.0;
sigma3 = 4.0;
H1 = fspecial('gaussian', [Radius,Radius], sigma1);
H2 = fspecial('gaussian', [Radius*2-1,Radius*2-1], sigma2);
H3 = fspecial('gaussian', [Radius*4-1,Radius*4-1], sigma3);
B1= imfilter(src, H1, 'replicate');
B2= imfilter(src, H2, 'replicate');
B3= imfilter(src, H3, 'replicate');
% figure;imshow(B3), title('B3');
D1=src-B1;
D2=B1-B2;
D3=B2-B3;
dest=(1-w1.*sign(D1)).*D1+w2*D2+w3*D3;
%%%%%%第二层%%%%%%%
src1=imresize(B3,0.5);
B4= imfilter(src1, H1, 'replicate');
B5= imfilter(src1, H2, 'replicate');
B6= imfilter(src1, H3, 'replicate');
D4=src1-B4;
D5=B4-B5;
D6=B5-B6;
dest1=(1-w1.*sign(D4)).*D4+w2*D5+w3*D6;
dest_1=imresize(dest1,2,'bilinear');
%%%%%%%%%%%第三层%%%%%%%%%5
src2=imresize(B6,0.5);
B7= imfilter(src2, H1, 'replicate');
B8= imfilter(src2, H2, 'replicate');
B9= imfilter(src2, H3, 'replicate');
D7=src2-B7;
D8=B7-B8;
D9=B8-B9;
dest2=(1-w1.*sign(D7)).*D7+w2*D8+w3*D9;
dest_2=imresize(dest2,4,'bilinear');
dest_1=imresize(dest_2,[size(src,1),size(src,2)]);
dest_2=imresize(dest_2,[size(src,1),size(src,2)]);
total=(1-w1.*sign(dest)).*dest+w2*dest_1+w3*dest_2;
%total=(1-0.5.*sign(dest)).*dest+0.3*dest_1+0.2*dest_2;
%removal2=total+src+sunlit  ;
removal2=src-0.3*total+sunlit ;
toc
imwrite(removal2,'baseline_multisharpen.tif')
figure,imshow(removal2,[]) 
im0316 = imread(url);
before_residual=removal2-im0316;
imwrite(before_residual,'before_residual.tif')
%%
w1=0.3;
w2=0.45;
w3=0.95;
dest=(1-w1.*sign(D1)).*D1+w2*D2+w3*D3+src;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%边界处理%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%jqronghe%%%%%%%%%%%%%%%%%%%%%%%%%%%%
im=imread(url);
tic
SE=strel('square',3);
for i=1:size(shadow,1)
    for j =1:size(shadow,2)
        if shadow(i,j)==0
            shadow1(i,j)=255;
        else
            shadow1(i,j)=0;
        end
    end
end
          
im_shadow_imdilate=imdilate(shadow1,SE); 
bj_shadow=im_shadow_imdilate-shadow1;
bj_shadow=bj_shadow/255;
im6=uint8(zeros(size(im,1),size(im,2),3));
for i=1:size(im_sunlit,1)
    for j=1:size(im_sunlit,2)
        for k=1:size(im_sunlit,3)
        if bj_shadow(i,j)==1
            im6(i,j,k)=bj_shadow(i,j)*im_sunlit(i,j,k);      
        end
        end
    end
end
% figure,imshow(im6) 

im_shadow_imerode=imerode(shadow1,SE);
bj_shadow1=shadow1-im_shadow_imerode;
bj_shadow1=bj_shadow1/255;
im7=uint8(zeros(size(im,1),size(im,2),3));
for i=1:size(im6,1)
    for j=1:size(im6,2)
        for k=1:size(im6,3)
        if bj_shadow1(i,j)==1
            im7(i,j,k)=bj_shadow1(i,j)*im(i,j,k);      
        end
        end
    end
end
figure,imshow(im7)   

im_shadowbj=im6+im7;
% figure,imshow(im_shadowbj)
% imwrite(im_shadowbj,'im_shadowbj.tif')
bj=bj_shadow+bj_shadow1;
% test_im = removal2;
% marker_bj = zeros(size(removal2, 1), size(removal2, 2));
%     [m,n,k]=size(removal2);
%     for i=1:m
%         for j=1:n
%             top=shadow(max(1,i-1),j);
%             bottom=shadow(min(m,i+1),j);
%             left=shadow(i,max(1,j-1));
%             right=shadow(i,min(n,j+1));
%             if ~(top==bottom && bottom==left && left==right)
%                 marker_bj(i,j)=1;
%             end
%         end
%     end
% im_bianjie=res;
% for i=6:size(im,1)-5
%     for j=6:size(im,2)-5
%         for k=1:size(im,3)
%             if marker_bj(i,j)==1               
%                 u=abs(im(i,j-3,k)-im(i,j+3,k))/im(i,j,k);
%                 u1=abs(im(i-3,j,k)-im(i+3,j,k))/im(i,j,k);
%                 umean=(u+u1);
%                 im_bianjie(i,j,k)=(umean)*im(i,j,k);
%             end
%         
%         end
%     end
% end
% % figure,imshow(im_bianjie)
% toc               
% test_im=removal2;
% h = fspecial('gaussian',5,1.5);
% pattern = imfilter(test_im, h);
% for i = 1:segnum
%     if label(i) == 0      
%         for ch = 1:3
%             fig = test_im(:,:,ch);            
%             for x = 3:size(fig, 1) - 2
%                 for y = 3:size(fig,2) - 2                    
%                     if seg(x,y) == i && seg(x-2, y) ~= i && marker_bj(x,y) == 1
%                         %if abs(test_im(x,y,3) - avg(i,3)) > 0.05
%                             fig(x,y) = pattern(x-2,y,ch);
%                             fig(x-2,y) = pattern(x,y,ch);
%                         %end
%                         %
%                     elseif seg(x,y) == i && seg(x+2, y) ~= i && marker_bj(x,y) == 1
%                         %if abs(test_im(x,y,3) - avg(i,3)) > 0.05
%                             fig(x,y) = pattern(x+2,y,ch);
%                             fig(x+2,y) = pattern(x,y,ch);
%                         %end
%                         %
%                     elseif seg(x,y) == i && seg(x, y+2) ~= i && marker_bj(x,y) == 1
%                         %if abs(test_im(x,y,3) - avg(i,3)) > 0.05
%                             fig(x,y) = pattern(x,y+2,ch);
%                             fig(x,y+2) = pattern(x,y,ch);
%                         %end
%                         %
%                     elseif seg(x,y) == i && seg(x, y-2) ~= i && marker_bj(x,y) == 1
%                         %if abs(test_im(x,y,3) - avg(i,3)) > 0.05
%                             fig(x,y) = pattern(x,y-2,ch);
%                             fig(x,y-2) = pattern(x,y,ch);
%                         %end
%                         %
%                     end
%                 end
%             end
%             test_im(:,:,ch) = fig;
%         end
%     end
%     
% end
% % figure,imshow(test_im);        
% imwrite(test_im,'guassbj.tif')
% %%%%%%%%%%%%%第二次%%%%%%%%%%%%%%%%
% test1_im=test_im;
% h = fspecial('average', [5 5]);
% pattern = imfilter(test1_im, h);
% for i = 1:segnum
%     if label(i) == 0      
%         for ch = 1:3
%             fig = test1_im(:,:,ch);            
%             for x = 3:size(fig, 1) - 2
%                 for y = 3:size(fig,2) - 2                    
%                     if seg(x,y) == i && seg(x-2, y) ~= i && marker_bj(x,y) == 1
%                         %if abs(test_im(x,y,3) - avg(i,3)) > 0.05
%                             fig(x,y) = pattern(x-2,y,ch);
%                             fig(x-2,y) = pattern(x,y,ch);
%                         %end
%                         %
%                     elseif seg(x,y) == i && seg(x+2, y) ~= i && marker_bj(x,y) == 1
%                         %if abs(test_im(x,y,3) - avg(i,3)) > 0.05
%                             fig(x,y) = pattern(x+2,y,ch);
%                             fig(x+2,y) = pattern(x,y,ch);
%                         %end
%                         %
%                     elseif seg(x,y) == i && seg(x, y+2) ~= i && marker_bj(x,y) == 1
%                         %if abs(test_im(x,y,3) - avg(i,3)) > 0.05
%                             fig(x,y) = pattern(x,y+2,ch);
%                             fig(x,y+2) = pattern(x,y,ch);
%                         %end
%                         %
%                     elseif seg(x,y) == i && seg(x, y-2) ~= i && marker_bj(x,y) == 1
%                         %if abs(test_im(x,y,3) - avg(i,3)) > 0.05
%                             fig(x,y) = pattern(x,y-2,ch);
%                             fig(x,y-2) = pattern(x,y,ch);
%                         %end
%                         %
%                     end
%                 end
%             end
%             test1_im(:,:,ch) = fig;
%         end
%     end
%     
% end
% figure,imshow(test1_im);  
% imwrite(test1_im,'averagebj.tif')
tic
im_bianjie=removal2;
for i=6:size(im,1)-5
    for j=6:size(im,2)-5
        for k=1:size(im,3)
            if bj(i,j)==1
                if shadow(i,j-5)==255
                    im_bianjie(i,j,k)=0;
                for m=-3:1
                    for n=-1:1
                %im_bianjie(i,j,k)=0.69*((abs(m)+abs(n))/9)*removal2(i+n,j+m,k)+ im_bianjie(i,j,k);
                im_bianjie(i,j,k)=0.51*((abs(m)+abs(n))/15)*removal2(i+n,j+m,k)+ im_bianjie(i,j,k);
                    end
                end                        
                else
                 im_bianjie(i,j,k)=0;   
                for m=-1:3
                    for n=-1:1
                %im_bianjie(i,j,k)=0.69*((abs(m)+abs(n))/9)*removal2(i+n,j+m,k)+ im_bianjie(i,j,k);
                im_bianjie(i,j,k)=0.51*((abs(m)+abs(n))/15)*removal2(i+n,j+m,k)+ im_bianjie(i,j,k);
                    end
                end      
               
                end
            end
        
        end
    end
end
toc
figure,imshow(im_bianjie)

imwrite(im_bianjie,'baseline_boundary.tif')

residual=im_bianjie-im;
figure,imshow(residual)
imwrite(residual,'residual.tif')
% guass_residual=test_im-im;
% imwrite(guass_residual,'gauss_residual.tif')
% ave_residual=test1_im-im;
% imwrite(ave_residual,'ave_residual.tif')