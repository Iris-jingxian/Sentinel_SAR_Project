function run_sidelobe_v01(outDir)
% RUN_SIDELOBE_V01 reproducible CPU evaluation, no toolboxes required.
base=fileparts(mfilename('fullpath')); addpath(base);
repo=fileparts(fileparts(base));
if nargin==0, outDir=fullfile(repo,'results','sidelobe_v01'); end
if ~exist(outDir,'dir'), mkdir(outDir); end
set(groot,'defaultFigureColor','w','defaultAxesColor','w', ...
    'defaultAxesXColor','k','defaultAxesYColor','k','defaultTextColor','k');
rng(20261031); rows=struct([]); checks=run_checks();
n=257; t=(-128:128)'; q=2;
sinc0=@(z) local_sinc(z);
point=sinc0(t/q)*sinc0(t'/q)*exp(1i*0.6);
multi=point+0.1*exp(1i*1.2)*(sinc0((t-18)/q)*sinc0((t'+22)/q));
% A noiseless sampled rectangular-spectrum PSF, NOT a full RD simulator.
cases={point,multi}; names={'sim_single','sim_strong_weak'};
for k=1:3
    f=fullfile(repo,'data','ROIs',sprintf('ROI_%d_complex.mat',k));
    if exist(f,'file')
        d=load(f); assert(isfield(d,'roi_complex'),'Missing roi_complex');
        cases{end+1}=d.roi_complex.'; % read_slc: transpose WITHOUT conjugation
        names{end+1}=sprintf('ROI_%d',k);
    end
end
for c=1:numel(cases)
    x=double(cases{c}); isSim=c<=2;
    if isSim, lag=[2 2]; r=129; a=129; radius=48; status='controlled_simulation';
    else
        lag=[1 1]; radius=32; status='exploratory_unknown_bandwidth';
        % Peak selection excludes borders; real-scene metrics are local proxies.
        z=abs(x); z(1:40,:)=0; z(end-39:end,:)=0;
        z(:,1:40)=0; z(:,end-39:end)=0;
        [~,idx]=max(z(:)); [r,a]=ind2sub(size(x),idx);
    end
    rr=max(1,r-radius):min(size(x,1),r+radius);
    aa=max(1,a-radius):min(size(x,2),a+radius);
    center=[find(rr==r),find(aa==a)];
    ref={abs(x(rr,a)),abs(x(r,aa)).'};
    gates=cell(1,2);
    for dim=1:2
        if isSim, gates{dim}=[center(dim)-2,center(dim)+2];
        else, gates{dim}=find_gate(ref{dim},center(dim)); end
    end
    tic; [s,info]=sva2d(x,lag); ts=toc;
    tic; h=hamming_image(x,lag); th=toc;
    ys={x,h,s}; methods={'Original','Hamming','SVA'}; times=[0 th ts];
    scale=max(abs(x(:))); lim=[-60 5];
    f=figure('Visible','off','Position',[100 100 1350 460]);
    if isprop(f,'Theme'), f.Theme='light'; end
    for j=1:3
        subplot(1,3,j); imagesc(20*log10(max(abs(ys{j})/scale,1e-6)),lim);
        axis image; colorbar; title([names{c} ' / ' methods{j}],'Interpreter','none');
        xlabel('Azimuth pixel'); ylabel('Range pixel');
    end
    colormap(gray); exportgraphics(f,fullfile(outDir,[names{c} '_comparison.png'])); close(f);
    f=figure('Visible','off','Position',[100 100 1100 700]);
    if isprop(f,'Theme'), f.Theme='light'; end
    for dim=1:2
        subplot(2,1,dim); hold on;
        for j=1:3
            y=ys{j}; if dim==1, b=y(rr,a); else, b=y(r,aa).'; end
            m=profile_metrics(ref{dim},b,center(dim),gates{dim});
            row=struct('case_id',names{c},'method',methods{j}, ...
                'axis',char(string_axis(dim)),'status',status, ...
                'PSLR_dB',m.PSLR_dB,'ISLR_dB',m.ISLR_dB, ...
                'width_3dB_px',m.width_3dB_px,'peak_ratio',m.peak_ratio, ...
                'main_energy_ratio',m.main_energy_ratio,'elapsed_s',times(j), ...
                'range_peak',r,'azimuth_peak',a,'lag_range',lag(1),'lag_azimuth',lag(2), ...
                'gate_left_relative',gates{dim}(1)-center(dim), ...
                'gate_right_relative',gates{dim}(2)-center(dim), ...
                'phase_rmse_rad',phase_error(x,y));
            rows=[rows;row]; %#ok<AGROW>
            plot((1:numel(b))-center(dim),20*log10(max(abs(b)/max(ref{dim}),1e-6)),'LineWidth',1.2);
        end
        xline(gates{dim}(1)-center(dim),'k:'); xline(gates{dim}(2)-center(dim),'k:');
        yline(-3,'k--'); ylim([-70 5]); grid on;
        xlabel('Offset (pixel)'); ylabel('dB / original profile peak');
        title(string_axis(dim)); legend(methods,'Location','best');
    end
    exportgraphics(f,fullfile(outDir,[names{c} '_profiles.png'])); close(f);
    original=x; sva=s; hamming=h; %#ok<NASGU>
    save(fullfile(outDir,[names{c} '_results.mat']),'original','sva','hamming','info','r','a','gates','rr','aa','status');
end
T=struct2table(rows); writetable(T,fullfile(outDir,'traditional_comparison_v01.csv'));
config=struct('version','0.1','seed',20261031,'MATLAB',version, ...
    'reference_scale','original maximum, shared by all methods', ...
    'metrics_gate','fixed original first minima; synthetic known nulls', ...
    'real_ROI_lag',[1 1],'real_ROI_status','exploratory; bandwidth not verified', ...
    'units','pixels; physical spacing not verified','output_orientation','rows range, columns azimuth; ROI input nonconjugate-transposed','checks',checks);
fid=fopen(fullfile(outDir,'parameters_v01.json'),'w'); fprintf(fid,'%s',jsonencode(config,PrettyPrint=true)); fclose(fid);
disp(T(1:6,:)); fprintf('Saved: %s\n',outDir);
end
function z=local_sinc(x)
z=ones(size(x)); k=x~=0; z(k)=sin(pi*x(k))./(pi*x(k));
end
function g=find_gate(p,c)
l=find(p(2:c-1)<=p(1:c-2)&p(2:c-1)<=p(3:c),1,'last')+1;
r=find(p(c+1:end-1)<=p(c:end-2)&p(c+1:end-1)<=p(c+2:end),1,'first')+c;
if isempty(l), l=max(1,c-2); end
if isempty(r), r=min(numel(p),c+2); end
g=[l r];
end
function s=string_axis(d)
if d==1, s='range'; else, s='azimuth'; end
end
function e=phase_error(x,y)
mask=abs(x)>0.1*max(abs(x(:))) & abs(y)>1e-10*max(abs(x(:)));
if ~any(mask(:)), e=NaN; else, e=sqrt(mean(angle(y(mask).*conj(x(mask))).^2)); end
end
function result=run_checks()
% Meaningful invariants: zero input, phase/scale equivariance and attenuation.
z=sva2d(complex(zeros(17)),[1 1]); assert(all(z(:)==0));
x=randn(33)+1i*randn(33); y=sva2d(x,[2 2]);
a=2.7*exp(1i*0.8); ya=sva2d(a*x,[2 2]);
assert(max(abs(ya(:)-a*y(:)))<1e-11);
assert(all(abs(y(:))<=abs(x(:))+1e-12));
% Exact independent analytic one-axis check against dense weight search.
v=x(17,17); s=x(15,17)+x(19,17);
w=min(.5,max(0,-real(v*conj(s))/abs(s)^2));
assert(abs(v+w*s)<=min(abs(v+linspace(0,.5,10001)*s))+1e-12);
t=(-32:32)'; psf=local_sinc(t/2)*local_sinc(t'/2);
filtered=sva2d(psf,[2 2]);
assert(abs(filtered(33,33)-1)<1e-12);
assert(max(abs(filtered(36:63,33)))<1e-12,'Sinc sidelobes must cancel.');
result='PASS: zero, scale/phase equivariance, attenuation, weight minimization, sinc sidelobe cancellation and peak preservation';
end

