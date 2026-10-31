function [y,info] = sva2d(x,lag)
% Complex joint-IQ SVA. Rows=range, columns=azimuth.
% lag = integer sampling/bandwidth ratio per axis; centered rectangular spectrum.
% Noninteger ratios or already weighted SLC require separate validation.
% Zero extension; outer lag pixels in each pass are left unchanged.
arguments
    x (:,:) {mustBeNumeric,mustBeFinite}
    lag (1,2) double {mustBeInteger,mustBePositive} = [1 1]
end
assert(~isempty(x) && all(size(x)>2*lag),'Input too small for lag.');
y=double(x);
for dim=1:2
    s=zeros(size(y),'like',y); k=lag(dim);
    if dim==1
        idx=(1+k):(size(y,1)-k);
        s(idx,:)=y(idx-k,:)+y(idx+k,:);
    else
        idx=(1+k):(size(y,2)-k);
        s(:,idx)=y(:,idx-k)+y(:,idx+k);
    end
    den=abs(s).^2; w=zeros(size(y)); valid=den>0;
    w(valid)=-real(y(valid).*conj(s(valid)))./den(valid);
    w=min(0.5,max(0,w));
    y=y+w.*s;
end
info=struct('version','0.1','variant','complex_joint_IQ_sequential', ...
    'lag',lag,'axis_order','range_then_azimuth','boundary','unchanged_outer_lag_per_pass');
end
