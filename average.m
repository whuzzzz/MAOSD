
function mean_h=average(sunlitR)
hist1 = imhist(sunlitR); 
hist1=hist1(2:256);
count=sum(hist1);
total=0;
for i=1:size(hist1,1)
    total = (i*hist1(i,1))+total;    
end
mean_h=total/count;
end