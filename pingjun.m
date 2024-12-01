function x_mean=pingjun(seg_shadow,segnum_shadow)
for k=1:segnum_shadow
x_total=0;
y_total=0;
count=0;
for i=1:size(seg_shadow,1)
for j=1:size(seg_shadow,2)
    if seg_shadow(i,j)==k
    x_total=i+x_total;
    y_total=j+y_total;
    count=count+1;
    end
end
end
x_mean(k,1)=round(x_total/count);
x_mean(k,2)=round(y_total/count);
end