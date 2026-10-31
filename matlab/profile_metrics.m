function m=profile_metrics(a,b,peak,bounds)
% Fixed reference first-null mainlobe gate for reproducible comparison.
% PSLR=20log10(max(side amplitude)/max(main amplitude)).
% ISLR=10log10(sum(side power)/sum(main power)).
a=abs(a(:)); b=abs(b(:)); n=numel(b);
main=false(n,1); main(bounds(1):bounds(2))=true;
peakOut=max(b(main)); peakIn=max(a(main));
m.PSLR_dB=20*log10(max(max(b(~main)),realmin)/max(peakOut,realmin));
m.ISLR_dB=10*log10(max(sum(b(~main).^2),realmin)/max(sum(b(main).^2),realmin));
m.width_3dB_px=width3(b,peak,bounds);
m.peak_ratio=peakOut/max(peakIn,realmin);
m.main_energy_ratio=sum(b(main).^2)/max(sum(a(main).^2),realmin);
m.gate_left=bounds(1); m.gate_right=bounds(2);
end
function w=width3(b,peak,bounds)
[pk,j]=max(b(bounds(1):bounds(2))); j=j+bounds(1)-1;
t=pk/sqrt(2); l=find(b(1:j)<t,1,'last');
r=find(b(j:end)<t,1,'first'); w=NaN;
if isempty(l)||isempty(r), return; end
r=r+j-1;
left=l+(t-b(l))/(b(l+1)-b(l));
right=r-1+(t-b(r-1))/(b(r)-b(r-1));
w=right-left;
end
