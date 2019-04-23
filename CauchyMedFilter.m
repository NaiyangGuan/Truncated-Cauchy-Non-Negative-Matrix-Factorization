function P=CauchyMedFilter(Q,im_h,im_w)
P=Q;
for num=1:size(Q,2),
    I=reshape(Q(:,num),im_h,im_w)*255;
    I=medfilt2(I);
    P(:,num)=reshape(double(I)/255,im_h*im_w,1);
end
return;