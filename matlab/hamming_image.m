function y = hamming_image(x,lag)
% Unit coherent-gain cosine-on-pedestal Hamming comparator.
% At integer oversampling lag, response on occupied band is
% (0.54+0.46*cos(2*pi*lag*f))/0.54. Zero boundaries, same as SVA.
y=double(x);
for dim=1:2
    k=lag(dim); s=zeros(size(y),'like',y);
    if dim==1
        idx=(1+k):(size(y,1)-k); s(idx,:)=y(idx-k,:)+y(idx+k,:);
    else
        idx=(1+k):(size(y,2)-k); s(:,idx)=y(:,idx-k)+y(:,idx+k);
    end
    y=y+(0.23/0.54)*s;
end
end
