% =========================================================================
%           On the Missing Area of the Bullet Cluster
%                         Brahim Benaissa
%                   https://Justpeers.com
%
%
%           Based on the Baryonic model of Famaey 2026  
%                       arXiv:2605.10022 
% =========================================================================

clear; close all; clc;

%% ------------------------------------------------------------------------
%  Section 1: CONFIGURATION 
% ------------------------------------------------------------------------
% --- Mezzi Parameters ---
Compliance            = 377;               % Mezzi's Compliance Constant 

% --- Lensing Smoothing ---
Sigma_smoothing       = false;     % set "true" to apply Lensing Smoothing     
smoothing_fwhm_arcsec = 10;        % 0 = No smoothing     

% --- Fundamental Units ---
G     = 6.67430e-11;
c     = 2.99792458e8;
pc    = 3.085677581491367e16; 
kpc   = 1e3 * pc;
Mpc   = 1e6*pc;
Msun  = 1.98847e30;
arcsec_to_kpc = 4.413;       % Rihtaršič 2026 


% --- Simulation Grid & Box Parameters ---
Box_3d                = 40 * Mpc;           % 3D box size in meters
N_2d                  = 5000;               %  2D adaptive grid resolution
N_z                   = 2000;               %  Line of sight grid resolution

% --- Dual-focus adaptive grid using inverse CDF ---   
stretch_factor        = 5;                 % Adaptive grid stretching factor, 0 = No Strech
focus_width           = 200e3 * pc;        % High-res focus width for 2D grid (x,y)
focus_width_z         = 40e3 * pc;         % High-res focus width for z-axis

% --- FFT / Uniform Proxy Grid Parameters ---
N_fft                 = N_2d;              % Match primary grid dimensions
fft_box_half_size_kpc = 4000;              % Half-size of the uniform FFT box in kpc

% --- Ray Tracing Parameters ---
N_theta               = 30;                % Number of polar angles for rays
N_phi                 = 60;                % Number of azimuthal angles for rays
N_r_ray               = 300;               % Number of radial steps along each ray


%% ------------------------------------------------------------------------
%  Section 2: BULLET CLUSTER MASS MODEL
% ------------------------------------------------------------------------
 
%% ------------------------------------------------------------------------
%  2.1:   COSMOLOGY
% ------------------------------------------------------------------------

H0 = 70.0;  Omega_m = 0.30;  Omega_L = 0.70;
E_z = @(z) sqrt(Omega_m*(1+z).^3 + Omega_L);
 
% Cosmology (Famaey 2026 places all sources at infinity)
z_lens = 0.296;

% Angular diameter distance to the lens
E_z_lens = sqrt(Omega_m*(1+z_lens)^3 + Omega_L);
D_C_lens = (c/1000)/H0 * trapz(linspace(0,z_lens,2000), 1./sqrt(Omega_m*(1+linspace(0,z_lens,2000)).^3 + Omega_L));
D_L  = D_C_lens/(1+z_lens)*Mpc;

% Critical surface density (match Famaey 2026)
Sigma_crit = 1827;   % 1.827e9 Msun / kpc^2 = 1827 Msun / pc^2
 
%% ------------------------------------------------------------------------
%  2.2: LOAD CATALOG
%% ------------------------------------------------------------------------

cat_filename = 'cluster_members_specz_cat.dat';

if ~isfile(cat_filename)
    error(['File not found: %s\n', 'Obtain full catalog from Rihtaršič+2026 data release.'], cat_filename);
end

fprintf('[LOAD] Reading %s...\n', cat_filename);
fid = fopen(cat_filename, 'r');
data = textscan(fid, '%d %f %f %f %f %s', 'HeaderLines', 1, 'CommentStyle', '#', 'MultipleDelimsAsOne', true);
fclose(fid);

gal_id    = data{1};
gal_ra    = data{2};        % degrees
gal_dec   = data{3};        % degrees
gal_mag   = data{4};        % F277W AB magnitude

N_cat = length(gal_id);
fprintf('[LOAD] %d cluster members loaded.\n', N_cat);

%% ------------------------------------------------------------------------
%  2.3: BCG IDENTIFICATION
%% ------------------------------------------------------------------------

[~, sort_mag] = sort(gal_mag);

% BCG1: brightest galaxy, defines coordinate origin
bcg1_idx = sort_mag(1);
RA_BCG1  = gal_ra(bcg1_idx);
Dec_BCG1 = gal_dec(bcg1_idx);
cos_dec  = cosd(Dec_BCG1);

% Convert ALL catalog positions to Famaey frame: x=west, y=north, in kpc
x_cat = -(gal_ra - RA_BCG1) .* 3600 .* cos_dec .* arcsec_to_kpc;
y_cat = (gal_dec - Dec_BCG1) .* 3600 .* arcsec_to_kpc;

% BCG3: brightest galaxy near expected (820, 220) kpc
dist_bcg3_exp = sqrt((x_cat - 820).^2 + (y_cat - 220).^2);
dist_bcg3_exp(bcg1_idx) = inf;
[~, bcg3_idx] = min(dist_bcg3_exp);

% BCG2: brightest galaxy near expected (106, 129) kpc
dist_bcg2_exp = sqrt((x_cat - 106).^2 + (y_cat - 129).^2);
dist_bcg2_exp([bcg1_idx, bcg3_idx]) = inf;
[~, bcg2_idx] = min(dist_bcg2_exp);

fprintf('[BCG] Identified from catalog:\n');
fprintf('      BCG1 (ID %d): mag=%.2f, pos=(%7.1f, %7.1f) kpc\n', gal_id(bcg1_idx), gal_mag(bcg1_idx), x_cat(bcg1_idx), y_cat(bcg1_idx));
fprintf('      BCG2 (ID %d): mag=%.2f, pos=(%7.1f, %7.1f) kpc [exp: ~106, 129]\n', gal_id(bcg2_idx), gal_mag(bcg2_idx), x_cat(bcg2_idx), y_cat(bcg2_idx));
fprintf('      BCG3 (ID %d): mag=%.2f, pos=(%7.1f, %7.1f) kpc [exp: ~820, 220]\n', gal_id(bcg3_idx), gal_mag(bcg3_idx), x_cat(bcg3_idx), y_cat(bcg3_idx));

% BCG properties (Famaey Section 2.3)
M_bcg  = 1.0e12 * Msun;
r_bcg  = 10e3 * pc;

%% ------------------------------------------------------------------------
%  2.4: MASS BUDGET
%% ------------------------------------------------------------------------

M_main_smooth = 1.196e13 * Msun;   % main galaxy component (incl. BCG1, BCG2)
M_sub_smooth  = 4.0e12  * Msun;   % subcluster galaxy component (incl. BCG3)

M_remain_main = M_main_smooth - 2*M_bcg;   % BCG1 + BCG2
M_remain_sub  = M_sub_smooth  - 1*M_bcg;   % BCG3

N_main_total = 166;   % non-BCG main cluster galaxies
N_sub_total  = 50;    % non-BCG subcluster galaxies

% Uniform mass per galaxy
gal_mass_uniform = M_remain_main / N_main_total;  % = 6.0e10 Msun by construction

assert(abs(gal_mass_uniform - M_remain_sub/N_sub_total) < 1e-6 * gal_mass_uniform, ...
    'Mass budget inconsistent: main=%.6e, sub=%.6e', gal_mass_uniform, M_remain_sub/N_sub_total);

fprintf('[MASS] Uniform galaxy mass: %.4e Msun (= %.2f x 10^10)\n', gal_mass_uniform/Msun, gal_mass_uniform/Msun/1e10);

%% ------------------------------------------------------------------------
%  2.5: PLUMMER PARAMETERS
%% ------------------------------------------------------------------------

a_main = 470e3 * pc;   % smooth model Plummer radius, main component
a_sub  = 245e3 * pc;   % smooth model Plummer radius, subcluster
gal_rs = 3e3 * pc;     % all non-BCG galaxies

%% ------------------------------------------------------------------------
%  2.6: INITIALIZE OUTPUT STRUCT
%% ------------------------------------------------------------------------

components = struct('name', {}, 'x_c', {}, 'y_c', {}, 'z_c', {}, 'type', {}, 'M_total', {}, 'r_s', {});

% --- GAS COMPONENTS (No-Taper: Flat Offset Profile) ---
components(1) = struct('name','Main Gas',         'x_c',190e3*pc, 'y_c',90e3*pc,  'z_c',0, 'type','Plummer', 'M_total',2.0e14*Msun,  'r_s',565e3*pc);
components(2) = struct('name','Subcluster Gas 1', 'x_c',525e3*pc, 'y_c',120e3*pc, 'z_c',0, 'type','Plummer', 'M_total',1.5e13*Msun,  'r_s',505e3*pc);
components(3) = struct('name','Subcluster Gas 2', 'x_c',620e3*pc, 'y_c',165e3*pc, 'z_c',0, 'type','Plummer', 'M_total',4.1e12*Msun,  'r_s',100e3*pc);
components(4) = struct('name','Sub Gas 3',        'x_c',720e3*pc, 'y_c',175e3*pc, 'z_c',0, 'type','Plummer', 'M_total',4.1e12*Msun,  'r_s',100e3*pc);

% --- BCGs ---
components(5) = struct('name','BCG 1', 'x_c',x_cat(bcg1_idx)*kpc, 'y_c',y_cat(bcg1_idx)*kpc, 'z_c',0, 'type','Plummer', 'M_total',M_bcg, 'r_s',r_bcg);
components(6) = struct('name','BCG 2', 'x_c',x_cat(bcg2_idx)*kpc, 'y_c',y_cat(bcg2_idx)*kpc, 'z_c',0, 'type','Plummer', 'M_total',M_bcg, 'r_s',r_bcg);
components(7) = struct('name','BCG 3', 'x_c',820*kpc, 'y_c',220*kpc, 'z_c',0, 'type','Plummer', 'M_total',M_bcg, 'r_s',r_bcg);

% Subcluster centre for coordinate transforms (Hardcoded to match Famaey 2026)  
x_sub_c_kpc = 820;
y_sub_c_kpc = 220;

%% =====================================================================
%  2.7: MODE-SPECIFIC GALAXY GENERATION
% =====================================================================
fprintf('\n[MODE] === Famaey No-Taper (Flat Offset Profile) ===\n');

cx_sub = x_sub_c_kpc * kpc;
cy_sub = y_sub_c_kpc * kpc;

SNAPTARGETSMAINXY = [
            0.0,   10.0;    5.0,   22.0;   65.0,   39.2;   78.5,    9.0;  
           56.6,   80.1;  102.3,   29.1;   62.6,   90.1;   43.2,  117.6;  
           96.3,   96.8;   64.2,  123.5;   76.4,  121.8;    5.7,  -24.4;  
          -46.4,  -25.3;   38.2,  -41.2;  -59.5,   -4.2;   59.8,   -2.7;  
           68.4,  -25.7;   15.9,  -77.3;  -51.2,  -50.3;    4.0,  -88.1;  
          -19.9,  -94.5;    9.7, -108.2; -106.8,  -21.7;   30.2, -115.6; 
         -119.2,  -36.8;  -81.0,  -96.4;  107.6,   67.3;  -25.3, -126.1; 
           -9.7,  130.0; -136.6,   24.0;   94.3, -107.6;  142.2,   33.6; 
         -106.0,  111.4;  -76.2, -140.8;  147.5,  -70.2; -124.8,  118.7; 
          170.6,  -27.9;   33.5,  170.4;   98.0,  144.0;  165.0,   64.0; 
          141.0,  121.0;  117.9,  149.5;  187.2,   61.0;  179.2,   84.6; 
          150.4, -135.5;  178.6,  103.1;  192.0,   82.0;  152.0,  161.9; 
           87.6,  213.8;   20.6,  232.4;   41.1,  242.9;   -4.0,  261.8; 
          237.9,  128.3;  111.6,  254.1;  187.7,  223.4;  254.9,  151.9; 
          229.5,  213.8;  255.9,  268.2;  -82.4,  181.3;   25.0,  347.4; 
          161.3,  328.2
        ];

        SNAPTARGETSSUBXY = [
           820.0, 210.0;  770.3, 211.6;  846.0, 256.4;  777.9, 144.1; 
           864.3, 277.2;  863.5, 161.4;  738.5, 239.4;  904.2, 241.2; 
           731.0, 222.9;  910.5, 218.5;  750.0, 293.6;  894.3, 149.6; 
           804.2, 313.8;  851.7, 321.5;  739.9, 304.5;  755.2, 335.8; 
           696.2, 338.5;  648.3, 229.7;  653.7, 296.7;  631.9, 244.7; 
           688.8, 407.7;  510.2, 248.4
        ];
        
        N_t_main = size(SNAPTARGETSMAINXY, 1);
        N_t_sub  = size(SNAPTARGETSSUBXY, 1);
        fprintf('       Targets: %d main + %d sub = %d total\n', N_t_main, N_t_sub, N_t_main+N_t_sub);

        % Generate main cluster galaxies
        qxyz_main = [1.0, 0.35, 1.0]; PA_main = 50.0;
        [xs, ys, zs, r0] = sampleflattened_famaey(N_main_total, a_main, 0.0, 0.0, 42, qxyz_main, PA_main);
        [xs, ys, zs] = snaptotargets_famaey(xs, ys, zs, r0, SNAPTARGETSMAINXY * kpc, 0.0, 0.0);
        
        % Generate subcluster galaxies
        qxyz_sub = [1.0, 0.35, 1.0]; PA_sub = 0.0;
        [xs2, ys2, zs2, r02] = sampleflattened_famaey(N_sub_total, a_sub, cx_sub, cy_sub, 43, qxyz_sub, PA_sub);
        [xs2, ys2, zs2] = snaptotargets_famaey(xs2, ys2, zs2, r02, SNAPTARGETSSUBXY * kpc, cx_sub, cy_sub);
        
        % Assemble with uniform mass
        comp_idx = 7;
        for i = 1:N_main_total
            comp_idx = comp_idx + 1;
            components(comp_idx) = struct('name',sprintf('Main Gal %d',i), ...
                'x_c',xs(i), 'y_c',ys(i), 'z_c',zs(i), 'type','Plummer', 'M_total',gal_mass_uniform, 'r_s',gal_rs);
        end
        for i = 1:N_sub_total
            comp_idx = comp_idx + 1;
            components(comp_idx) = struct('name',sprintf('Sub Gal %d',i), ...
                'x_c',xs2(i), 'y_c',ys2(i), 'z_c',zs2(i), 'type','Plummer', 'M_total',gal_mass_uniform, 'r_s',gal_rs);
        end

%% =====================================================================
%  2.8: VERIFICATION & POST-PROCESSING
% =====================================================================

N_comp = length(components);
fprintf('\n[VERIFY] Total components: %d\n', N_comp);
fprintf('         Gas: 4 | BCGs: 3 | Galaxies: %d\n', N_comp - 7);

M_total_check = 0;
for i = 5:N_comp
    M_total_check = M_total_check + components(i).M_total;
end
M_expected = M_main_smooth + M_sub_smooth;

assert(abs(M_total_check - M_expected) < 1e-9 * M_expected, 'MASS CONSERVATION FAILED');
fprintf('[VERIFY] Mass conservation verified (diff=%.2e)\n', abs(M_total_check-M_expected)/M_expected);

star_idx = find(strcmp({components.type}, 'Plummer') & (contains({components.name}, 'Gal') | contains({components.name}, 'BCG')));
gas_idx  = find(strcmp({components.type}, 'Plummer') & contains({components.name}, 'Gas'));

comp_xy_kpc = zeros(N_comp, 2);
for j = 1:N_comp
    comp_xy_kpc(j,1) = components(j).x_c / kpc;
    comp_xy_kpc(j,2) = components(j).y_c / kpc;
end

names_comp = {components.name};
idx_bcg1 = find(strcmp(names_comp, 'BCG 1'));
idx_bcg2 = find(strcmp(names_comp, 'BCG 2'));
idx_bcg3 = find(strcmp(names_comp, 'BCG 3'));

%% =====================================================================
%  2.9: BRIDGE: MAP COMPONENTS TO PIPELINE MACRO-VARIABLES
% =====================================================================
fprintf('\n[BRIDGE] Mapping discrete components to pipeline variables...\n');

idx_main_gas = find(strcmp(names_comp, 'Main Gas'));
idx_sub_gas1 = find(strcmp(names_comp, 'Subcluster Gas 1'));
idx_sub_gas2 = find(strcmp(names_comp, 'Subcluster Gas 2'));
idx_sub_gas3 = find(strcmp(names_comp, 'Sub Gas 3'));

if isempty(idx_sub_gas2), idx_sub_gas2 = NaN; end
if isempty(idx_sub_gas3), idx_sub_gas3 = NaN; end

arcsec_conv = (180/pi) * 3600 / D_L;
comp_xy_arcsec = zeros(N_comp, 2);
for j = 1:N_comp
    comp_xy_arcsec(j, 1) = components(j).x_c * arcsec_conv;
    comp_xy_arcsec(j, 2) = components(j).y_c * arcsec_conv;
end

main_gal_xy = comp_xy_arcsec(idx_bcg1, :);
sub_gal_xy  = comp_xy_arcsec(idx_bcg3, :);
main_gas_xy = comp_xy_arcsec(idx_main_gas, :);
sub_gas1_xy = comp_xy_arcsec(idx_sub_gas1, :);

if ~isnan(idx_sub_gas2), sub_gas2_xy = comp_xy_arcsec(idx_sub_gas2, :); else, sub_gas2_xy = [NaN, NaN]; end
if ~isnan(idx_sub_gas3), sub_gas3_xy = comp_xy_arcsec(idx_sub_gas3, :); else, sub_gas3_xy = [NaN, NaN]; end

fprintf('[BRIDGE] Pipeline variables generated successfully.\n');

%% =====================================================================
%  2.10: VISUALIZATION
% =====================================================================
kpc_per_arcsec = arcsec_to_kpc;
xlim_arcsec = [-150, 300];
ylim_arcsec = [-150, 200];

%% =====================================================================
%    FIGURE 1: SIMPLE TOPOLOGY  
% =====================================================================
fig1 = figure('Position',[100 100 1000 800],'Color','w','Name','Fig1_Topology');
axes('Position',[0.08 0.08 0.88 0.88]); hold on; axis equal;
set(gca,'Color','w','XColor',[0.3 0.3 0.3],'YColor',[0.3 0.3 0.3]);

c_main_gas = [0.05 0.40 0.65];   c_sub_gas  = [0.60 0.80 0.30];   
c_bcg_main = [0.00 0.35 0.60];   c_bcg_sub  = [0.55 0.75 0.25];   
c_main_gal = [0.00 0.35 0.60];   c_sub_gal  = [0.55 0.75 0.25];

theta = linspace(0,2*pi,200);
for i = gas_idx
    x_c = comp_xy_kpc(i,1); y_c = comp_xy_kpc(i,2); r = components(i).r_s / kpc;
    if contains(components(i).name,'Main Gas')
        fill(x_c + r*cos(theta), y_c + r*sin(theta), c_main_gas, 'FaceAlpha',0.20,'EdgeColor','k','LineWidth',0.8,'HandleVisibility','off');
    elseif contains(components(i).name,'Subcluster Gas 1')
        fill(x_c + r*cos(theta), y_c + r*sin(theta), c_sub_gas, 'FaceAlpha',0.15,'EdgeColor','k','LineWidth',0.8,'HandleVisibility','off');
    elseif contains(components(i).name,'Subcluster Gas 2')
        fill(x_c + r*cos(theta), y_c + r*sin(theta), c_sub_gas, 'FaceAlpha',0.25,'EdgeColor','k','LineWidth',0.8,'HandleVisibility','off');
    elseif contains(components(i).name,'Sub Gas 3')
        fill(x_c + r*cos(theta), y_c + r*sin(theta), c_sub_gas, 'FaceAlpha',0.25,'EdgeColor','k','LineWidth',0.8,'HandleVisibility','off');
    end
end

r_bcg_plot = 20; 
for i = [idx_bcg1, idx_bcg2, idx_bcg3]
    x_c = comp_xy_kpc(i,1); y_c = comp_xy_kpc(i,2); 
    if i == idx_bcg3, bcg_color = c_bcg_sub; else, bcg_color = c_bcg_main; end
    fill(x_c + r_bcg_plot*cos(theta), y_c + r_bcg_plot*sin(theta), bcg_color, 'FaceAlpha',0.90,'EdgeColor','k','LineWidth',1.5,'HandleVisibility','off');
end

main_gal_idx = find(contains(names_comp,'Main Gal'));
for i = 1:length(main_gal_idx)
    plot(comp_xy_kpc(main_gal_idx(i),1), comp_xy_kpc(main_gal_idx(i),2), 'o', 'MarkerSize', 3, 'MarkerFaceColor', c_main_gal, 'MarkerEdgeColor', 'k', 'LineWidth', 0.4, 'HandleVisibility','off');
end

sub_gal_idx = find(contains(names_comp,'Sub Gal'));
for i = 1:length(sub_gal_idx)
    plot(comp_xy_kpc(sub_gal_idx(i),1), comp_xy_kpc(sub_gal_idx(i),2), 'o', 'MarkerSize', 3, 'MarkerFaceColor', c_sub_gal, 'MarkerEdgeColor', 'k', 'LineWidth', 0.4, 'HandleVisibility','off');
end

for i = gas_idx
    x_c = comp_xy_kpc(i,1); y_c = comp_xy_kpc(i,2); r = components(i).r_s / kpc;
    if contains(components(i).name,'Main Gas')
        text(x_c, y_c + r + 15, 'Main Gas', 'HorizontalAlignment','center','FontSize',9,'Color','k','FontWeight','bold','HandleVisibility','off');
    elseif contains(components(i).name,'Subcluster Gas 1')
        text(x_c, y_c + r + 15, 'Sub Gas 1', 'HorizontalAlignment','center','FontSize',9,'Color','k','FontWeight','bold','HandleVisibility','off');
    elseif contains(components(i).name,'Subcluster Gas 2')
        text(x_c + r + 25, y_c, 'Sub Gas 2', 'HorizontalAlignment','left','FontSize',9,'Color','k','FontWeight','bold','HandleVisibility','off');
    elseif contains(components(i).name,'Sub Gas 3')
        text(x_c - r - 25, y_c, 'Sub Gas 3', 'HorizontalAlignment','right','FontSize',9,'Color','k','FontWeight','bold','HandleVisibility','off');
    end
end

text(comp_xy_kpc(idx_bcg1,1), comp_xy_kpc(idx_bcg1,2)+r_bcg_plot+25, 'BCG1', 'HorizontalAlignment','center','FontSize',10,'Color','k','FontWeight','bold','HandleVisibility','off');
text(comp_xy_kpc(idx_bcg2,1), comp_xy_kpc(idx_bcg2,2)+r_bcg_plot+25, 'BCG2', 'HorizontalAlignment','center','FontSize',10,'Color','k','FontWeight','bold','HandleVisibility','off');
text(comp_xy_kpc(idx_bcg3,1), comp_xy_kpc(idx_bcg3,2)+r_bcg_plot+25, 'BCG3', 'HorizontalAlignment','center','FontSize',10,'Color','k','FontWeight','bold','HandleVisibility','off');

leg_handles = []; leg_labels = {};
h_gas_main = plot(NaN, NaN, 's', 'MarkerSize', 14, 'MarkerFaceColor', c_main_gas, 'MarkerEdgeColor', 'k', 'LineWidth', 0.8);
leg_handles(end+1) = h_gas_main; leg_labels{end+1} = 'Main Gas';
h_gas_sub  = plot(NaN, NaN, 's', 'MarkerSize', 12, 'MarkerFaceColor', c_sub_gas, 'MarkerEdgeColor', 'k', 'LineWidth', 0.8);
leg_handles(end+1) = h_gas_sub; leg_labels{end+1} = 'Sub Gas';
h_bcg = plot(NaN, NaN, 'o', 'MarkerSize', 10, 'MarkerFaceColor', c_bcg_main, 'MarkerEdgeColor', 'k', 'LineWidth', 1.2);
leg_handles(end+1) = h_bcg; leg_labels{end+1} = 'Main BCGs';
h_bcg_sub = plot(NaN, NaN, 'o', 'MarkerSize', 10, 'MarkerFaceColor', c_bcg_sub, 'MarkerEdgeColor', 'k', 'LineWidth', 1.2);
leg_handles(end+1) = h_bcg_sub; leg_labels{end+1} = 'Subcluster BCG';
h_gal_main = plot(NaN, NaN, 'o', 'MarkerSize', 6, 'MarkerFaceColor', c_main_gal, 'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
leg_handles(end+1) = h_gal_main; leg_labels{end+1} = 'Main Galaxies';
h_gal_sub = plot(NaN, NaN, 'o', 'MarkerSize', 6, 'MarkerFaceColor', c_sub_gal, 'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
leg_handles(end+1) = h_gal_sub; leg_labels{end+1} = 'Sub Galaxies';

legend(leg_handles, leg_labels, 'Location', 'northwest', 'FontSize', 9, 'Box', 'on', 'EdgeColor', [0.6 0.6 0.6]);

xlim([-600 1100]); ylim([-600 800]);
xlabel('x [kpc]','FontSize',12); ylabel('y [kpc]','FontSize',12);
grid on; set(gca,'GridAlpha',0.15);

bar_len = 100; bar_x = 900; bar_y = -500;
line([bar_x bar_x+bar_len], [bar_y bar_y], 'Color','k','LineWidth',2.5,'HandleVisibility','off');
line([bar_x bar_x], [bar_y-5 bar_y+5], 'Color','k','LineWidth',2,'HandleVisibility','off');
line([bar_x+bar_len bar_x+bar_len], [bar_y-5 bar_y+5], 'Color','k','LineWidth',2,'HandleVisibility','off');
text(bar_x+bar_len/2, bar_y+15, '100 kpc', 'HorizontalAlignment','center','FontSize',9,'FontWeight','bold', 'Color', 'k', 'HandleVisibility','off');

filename_base = 'Fig1_Topology';
savefig(fig1, [filename_base, '.fig']);
print(fig1, filename_base, '-djpeg', '-r300');
fprintf('[FIGURE 1] Saved successfully.\n');

%% =================================================================
%  FIGURE 2: COMPARISON PLOT
%% =================================================================
fprintf('\n[COMPARE] Generating multi-slice gas surface density comparison...\n');

match_gas_M = [2.0e14, 1.5e13, 1.5e13, -6.8e12] * Msun;
match_gas_a = [565, 505, 90, 70] * 1e3 * pc;
match_gas_x = [190, 525, 670, 670] * 1e3 * pc;
match_gas_y = [90, 120, 170, 170] * 1e3 * pc;

notaper_gas_M = [components(1).M_total, components(2).M_total, components(3).M_total, components(4).M_total];
notaper_gas_a = [components(1).r_s, components(2).r_s, components(3).r_s, components(4).r_s];
notaper_gas_x = [components(1).x_c, components(2).x_c, components(3).x_c, components(4).x_c];
notaper_gas_y = [components(1).y_c, components(2).y_c, components(3).y_c, components(4).y_c];

y_slices = [120, 170, 220]; 
slice_colors = [0.000, 0.447, 0.741; 0.850, 0.325, 0.098; 0.466, 0.674, 0.188];
x_slice_kpc = linspace(400, 900, 500);
X_m = x_slice_kpc * 1e3 * pc;

fig2 = figure('Name', 'Fig2_GasDensityComparison', 'Color', 'w', 'Position', [100, 100, 850, 650], 'PaperPositionMode', 'auto');
ax = axes('Parent', fig2); hold(ax, 'on');
c_ref = [0.3, 0.3, 0.3]; 

for s = 1:length(y_slices)
    y_val = y_slices(s);
    Y_m = y_val * 1e3 * pc * ones(size(X_m));
    Sigma_match = zeros(size(X_m));
    Sigma_notaper = zeros(size(X_m));

    for i = 1:length(match_gas_M)
        R2_match = (X_m - match_gas_x(i)).^2 + (Y_m - match_gas_y(i)).^2;
        Sigma_match = Sigma_match + (match_gas_M(i) * match_gas_a(i)^2) ./ (pi * (R2_match + match_gas_a(i)^2).^2);
        R2_notaper = (X_m - notaper_gas_x(i)).^2 + (Y_m - notaper_gas_y(i)).^2;
        Sigma_notaper = Sigma_notaper + (notaper_gas_M(i) * notaper_gas_a(i)^2) ./ (pi * (R2_notaper + notaper_gas_a(i)^2).^2);
    end

    Sigma_match_msun = Sigma_match * (pc^2) / Msun;
    Sigma_notaper_msun = Sigma_notaper * (pc^2) / Msun;

    semilogy(ax, x_slice_kpc, Sigma_match_msun, '-', 'Color', slice_colors(s, :), 'LineWidth', 2.5, 'DisplayName', sprintf('y = %d kpc (Famaey Model)', y_val));
    semilogy(ax, x_slice_kpc, Sigma_notaper_msun, '--', 'Color', slice_colors(s, :), 'LineWidth', 2.5, 'DisplayName', sprintf('y = %d kpc (Flat positive)', y_val));
end

xline(ax, 670, ':', 'Color', c_ref, 'LineWidth', 1.5, 'HandleVisibility', 'off', 'Label', 'Subcluster Gas Center (670 kpc)', 'LabelOrientation', 'horizontal', 'LabelVerticalAlignment', 'bottom');
xlabel(ax, 'X [kpc]', 'FontSize', 13, 'FontWeight', 'bold', 'Interpreter', 'latex');
ylabel(ax, 'Gas Surface Density $\Sigma$ [$M_\odot$ / pc$^2$]', 'FontSize', 13, 'FontWeight', 'bold', 'Interpreter', 'latex');
lgd = legend(ax, 'Location', 'northeast', 'FontSize', 10, 'Box', 'on', 'EdgeColor', [0.7 0.7 0.7]);
set(lgd, 'Interpreter', 'tex');

grid(ax, 'on');
set(ax, 'GridColor', [0.85 0.85 0.85], 'GridAlpha', 0.6, 'MinorGridColor', [0.92 0.92 0.92], 'MinorGridAlpha', 0.4, 'GridLineStyle', ':');
set(ax, 'FontSize', 12, 'FontName', 'Helvetica', 'LineWidth', 1.2, 'Box', 'off');

ann_text = {'{\bfModel Comparison:}', 'Solid lines: Famaey Model', '(uses negative mass taper, double peak)', '', 'Dashed lines: Flat positive model', '(uses 2 offset positive spheres,', 'creating a flat, wide peak)'};
text(ax, 420, 5e6, ann_text, 'Interpreter', 'tex', 'FontSize', 10, 'FontName', 'Helvetica', 'BackgroundColor', [0.98 0.98 0.98], 'EdgeColor', [0.6 0.6 0.6], 'Margin', 6, 'VerticalAlignment', 'top', 'HorizontalAlignment', 'left');
hold(ax, 'off');

filename_base = 'Fig2_GasDensityComparison';
savefig(fig2, [filename_base, '.fig']);
print(fig2, filename_base, '-djpeg', '-r300');
fprintf('[FIGURE 2] Multi-slice plot saved successfully.\n');

%% ------------------------------------------------------------------------
% SECTION 3: COMPUTATIONAL GRIDS (ADAPTIVE STRETCHED GRID)
% ------------------------------------------------------------------------

u_faces = linspace(-1, 1, N_2d + 1);
if stretch_factor == 0
    x_faces = u_faces * (Box_3d/2); y_faces = u_faces * (Box_3d/2);
else
    fprintf('[GRID] Building dual-focus adaptive grid (Main + Subcluster)...\n');
    x_faces = generate_dual_focus_faces(u_faces, Box_3d/2, 0, x_sub_c_kpc*kpc, focus_width);
    y_faces = generate_dual_focus_faces(u_faces, Box_3d/2, 0, y_sub_c_kpc*kpc, focus_width);
end

dx_fine = diff(x_faces); dy_fine = diff(y_faces); 
x_fine = (x_faces(1:end-1) + x_faces(2:end)) / 2;
y_fine = (y_faces(1:end-1) + y_faces(2:end)) / 2;

u_z_faces = linspace(-1, 1, N_z + 1);
if stretch_factor == 0
    z_faces = u_z_faces * (Box_3d/2);
else
    z_faces = generate_dual_focus_faces(u_z_faces, Box_3d/2, 0, 0, focus_width_z);
end

dz_3d = diff(z_faces);
z_axis = (z_faces(1:end-1) + z_faces(2:end)) / 2;

theta_x_faces = x_faces / D_L * (180/pi) * 3600;
theta_y_faces = y_faces / D_L * (180/pi) * 3600;
theta_x = (theta_x_faces(1:end-1) + theta_x_faces(2:end)) / 2;
theta_y = (theta_y_faces(1:end-1) + theta_y_faces(2:end)) / 2;

[X_FINE, Y_FINE] = ndgrid(x_fine, y_fine);

%% ------------------------------------------------------------------------
% SECTION 4: ANALYTICAL PLUMMER POTENTIALS
% ------------------------------------------------------------------------
fprintf('[POISSON] Building vectorized Plummer potential function...\n');

M_arr     = [components.M_total]';
xc_arr    = [components.x_c]';
yc_arr    = [components.y_c]';
zc_arr    = [components.z_c]';
a2_arr    = ([components.r_s]'.^2);

Phi_total_pos_func = @(x,y,z) sum( ...
    (-G * reshape(M_arr, 1, 1, [])) ./ sqrt( ...
        (x - reshape(xc_arr, 1, 1, [])).^2 + ...
        (y - reshape(yc_arr, 1, 1, [])).^2 + ...
        (z - reshape(zc_arr, 1, 1, [])).^2 + ...
        reshape(a2_arr, 1, 1, []) ...
    ), 3 ...
);

M_total_bary = sum(M_arr);
M_true_total_bary = M_total_bary; 
fprintf('  True total baryonic M: %.3e Msun\n', M_true_total_bary/Msun);

%% ------------------------------------------------------------------------
% SECTION 5: MEZZI RAY TRACING
% ------------------------------------------------------------------------
R_source_limit = max(D_L, 0.1*pc);
R_box = Box_3d/2;

fprintf('[K] Ray tracing directly on 2D planes...\n');

if R_source_limit <= R_box
    r_ray = logspace(log10(0.1*pc), log10(R_source_limit), N_r_ray)';
else
    N_r_int = round(N_r_ray * (log10(R_box)-log10(0.1*pc)) / (log10(R_source_limit)-log10(0.1*pc)));
    N_r_int = max(N_r_int,50);
    N_r_ext = N_r_ray - N_r_int;
    r_int = logspace(log10(0.1*pc), log10(R_box), N_r_int)';
    r_ext = logspace(log10(R_box), log10(R_source_limit), N_r_ext+1)';
    r_ray = [r_int; r_ext(2:end)];
end

sum_inv_zeta_global = zeros(size(X_FINE));
total_path_length = 0;

fprintf('[K] Starting line-of-sight ray tracing over %d Z-planes...\n', N_z);
tic;
for k = 1:N_z
    if mod(k, 20) == 0 || k == 1 || k == N_z
        fprintf('  [K] Processing Z-plane %d of %d (%.1f%%)...\n', k, N_z, 100*k/N_z);
    end
    
    Z_k = z_axis(k);
    Z_plane = Z_k * ones(size(X_FINE));
    
    [~, Zeta_2D] = trace_component_K( ...
        struct('x_c',0,'y_c',0,'z_c',0), Phi_total_pos_func, M_true_total_bary, ...
        X_FINE, Y_FINE, Z_plane, r_ray, R_box, c, Compliance, N_theta, N_phi);
    
    Zeta_2D = max(Zeta_2D, realmin);
    
    dz = dz_3d(k);
    sum_inv_zeta_global = sum_inv_zeta_global + (dz ./ Zeta_2D);
    total_path_length = total_path_length + dz;
end
fprintf('[K] Ray tracing complete. Elapsed time: %.2f seconds.\n\n', toc);

zeta_2D_fine = total_path_length ./ sum_inv_zeta_global;
fprintf('[K] Global K complete. Min zeta = %.4e\n', min(zeta_2D_fine(:)));

%% ------------------------------------------------------------------------
% SECTION 6: PROJECTION, RESCALING & LENSING PIPELINE
% ------------------------------------------------------------------------
fprintf('[LENS] Projecting densities and building convergence maps...\n');

Sigma_stars_fine_SI = zeros(size(X_FINE));
Sigma_gas_fine_SI   = zeros(size(X_FINE));

for j = 1:N_comp
    comp = components(j);
    if comp.M_total == 0, continue; end

    if strcmp(comp.type, 'Plummer')
        Xs_f = X_FINE - comp.x_c; Ys_f = Y_FINE - comp.y_c;
        R2_f = Xs_f.^2 + Ys_f.^2;
        Sigma_f = (comp.M_total .* comp.r_s^2) ./ (pi .* (R2_f + comp.r_s^2).^2);
        
        if ismember(j, star_idx)
            Sigma_stars_fine_SI = Sigma_stars_fine_SI + Sigma_f;
        else
            Sigma_gas_fine_SI = Sigma_gas_fine_SI + Sigma_f;
        end
    end
end

Sigma_stars_fine = Sigma_stars_fine_SI * (pc^2)/Msun;
Sigma_gas_fine   = Sigma_gas_fine_SI   * (pc^2)/Msun;

% MEZZI STEP 3: TRUE FRAME MASS VIA AREA & GEOMETRY
J_vol = 1 ./ zeta_2D_fine;        
J_vol(~isfinite(J_vol)) = 1; 
J_vol(J_vol < 0) = 0;        
J = J_vol.^(2/3);

Sigma_true_stars = Sigma_stars_fine .* J;
Sigma_true_gas   = Sigma_gas_fine .* J;

lambda_2D = (1 ./ zeta_2D_fine).^(2/3);
lambda_2D(~isfinite(lambda_2D)) = 1;

[DX, DY] = ndgrid(dx_fine, dy_fine);
dA_obs_pc2 = (DX/pc) .* (DY/pc); 

M_obs_stars = sum(Sigma_stars_fine(:) .* dA_obs_pc2(:));
M_obs_gas   = sum(Sigma_gas_fine(:)   .* dA_obs_pc2(:));
M_obs_total = M_obs_stars + M_obs_gas;

M_true_stars = sum(Sigma_true_stars(:) .* dA_obs_pc2(:));
M_true_gas   = sum(Sigma_true_gas(:)   .* dA_obs_pc2(:));
M_true_total = M_true_stars + M_true_gas;
mass_ratio_eta = M_true_total / M_obs_total;

fprintf('  Observed baryonic: %.3e Msun\n', M_obs_total);
fprintf('  True baryonic:     %.3e Msun (eta=%.3f)\n', M_true_total, mass_ratio_eta);

Sigma_obs_fine = Sigma_stars_fine + Sigma_gas_fine;
kappa_bary        = Sigma_obs_fine / Sigma_crit;
kappa_mass_reveal = (Sigma_true_stars + Sigma_true_gas) / Sigma_crit;
kappa_curv_amp    = lambda_2D .* kappa_bary;
 
%% ------------------------------------------------------------------------
% SECTION 7: LENSING SOLVER (UNIFORM PROXY GRID)
% ------------------------------------------------------------------------
fprintf('[LENS] Preparing uniform proxy grid for FFT...\n');
 
fft_box_half_size = fft_box_half_size_kpc * 1e3 * pc; 
x_uni = linspace(-fft_box_half_size, fft_box_half_size, N_fft);
y_uni = linspace(-fft_box_half_size, fft_box_half_size, N_fft);

fprintf('[FFT] Cropping uniform grid to +/-%d kpc (Resolution: %.2f kpc/pix)\n', fft_box_half_size_kpc, (2*fft_box_half_size_kpc)/N_fft);

theta_x_uni = x_uni / D_L * (180/pi) * 3600;
theta_y_uni = y_uni / D_L * (180/pi) * 3600;
dtheta_uni = theta_x_uni(2) - theta_x_uni(1);

models = struct();
models.bary.name        = 'Baryonic Baseline'; models.bary.kappa = kappa_bary;
models.mass_reveal.name = 'Mass Revelation';   models.mass_reveal.kappa = kappa_mass_reveal;
models.curv_amp.name    = 'Curvature Amplification'; models.curv_amp.kappa = kappa_curv_amp;

F_kappa_bary = griddedInterpolant({theta_x, theta_y}, kappa_bary, 'linear');
F_kappa_mr   = griddedInterpolant({theta_x, theta_y}, kappa_mass_reveal, 'linear');
F_kappa_ca   = griddedInterpolant({theta_x, theta_y}, kappa_curv_amp, 'linear');

kappa_bary_uni = F_kappa_bary({theta_x_uni, theta_y_uni});
kappa_mr_uni   = F_kappa_mr({theta_x_uni, theta_y_uni});
kappa_ca_uni   = F_kappa_ca({theta_x_uni, theta_y_uni});

if Sigma_smoothing && smoothing_fwhm_arcsec > 0
    sigma_arcsec = smoothing_fwhm_arcsec / 2.355;
    sigma_pix_uniform = sigma_arcsec / dtheta_uni;
    
    fprintf('[LENS] Applying physical PSF smoothing (FWHM=%.1f\", sigma=%.2f pix)...\n', smoothing_fwhm_arcsec, sigma_pix_uniform);
    tic;
    
    kappa_bary_uni = imgaussfilt(kappa_bary_uni, sigma_pix_uniform);
    kappa_mr_uni   = imgaussfilt(kappa_mr_uni, sigma_pix_uniform);
    kappa_ca_uni   = imgaussfilt(kappa_ca_uni, sigma_pix_uniform);
    
    fprintf('[LENS] Interpolating smoothed maps back to adaptive grid...\n');
    F_kappa_bary_smoothed = griddedInterpolant({theta_x_uni, theta_y_uni}, kappa_bary_uni, 'linear');
    F_kappa_mr_smoothed   = griddedInterpolant({theta_x_uni, theta_y_uni}, kappa_mr_uni, 'linear');
    F_kappa_ca_smoothed   = griddedInterpolant({theta_x_uni, theta_y_uni}, kappa_ca_uni, 'linear');
    
    kappa_bary        = F_kappa_bary_smoothed({theta_x, theta_y});
    kappa_mass_reveal = F_kappa_mr_smoothed({theta_x, theta_y});
    kappa_curv_amp    = F_kappa_ca_smoothed({theta_x, theta_y});
    
    fprintf('[LENS] Smoothing complete. Elapsed time: %.2f seconds.\n\n', toc);
end
 

%% ------------------------------------------------------------------------
% SECTION 8: VISUALIZATION
% ------------------------------------------------------------------------

%% =====================================================================
%  FIGURE 3: BARYONIC TOPOLOGY MAP
%% =====================================================================
c_main_gal  = [0.100 0.350 0.600]; c_sub_gal   = [0.050 0.250 0.450];
c_main_gas  = [0.850 0.300 0.100]; c_sub_gas   = [0.950 0.550 0.150];

LW = struct('thin',0.6,'med',1.0,'thick',1.8,'heavy',2.5);
MS = struct('s',6,'m',10,'l',14,'xl',20);

comp = struct();
for j = 1:N_comp
    if components(j).M_total == 0, continue; end
    comp(j).name   = components(j).name;
    comp(j).x_c    = components(j).x_c / D_L * (180/pi) * 3600;
    comp(j).y_c    = components(j).y_c / D_L * (180/pi) * 3600;
    comp(j).r_s    = components(j).r_s / D_L * (180/pi) * 3600;
    comp(j).M      = components(j).M_total / Msun;
    comp(j).is_gas = contains(components(j).name, 'Gas');
end

fig3 = figure('Position',[100 100 900 780],'Color','w','Name','Fig3_BaryonicTopologyInGrid','PaperPositionMode','auto');ax = axes('Position',[0.10 0.10 0.78 0.86]);
hold on;

set(gca,'Color',[0.98 0.98 0.98]); grid off;
set(gca,'GridColor',[0.90 0.90 0.90],'GridAlpha',0.5,'MinorGridColor',[0.94 0.94 0.94],'MinorGridAlpha',0.3,'FontSize',11,'Box','on','LineWidth',0.8);

all_x = [comp.x_c]; all_y = [comp.y_c]; all_r = [comp.r_s];
x_min_comp = min(all_x - all_r); x_max_comp = max(all_x + all_r);
y_min_comp = min(all_y - all_r); y_max_comp = max(all_y + all_r);

zoom_factor = 1.4;
x_cen = (x_min_comp + x_max_comp)/2; y_cen = (y_min_comp + y_max_comp)/2;
x_half = (x_max_comp - x_min_comp)/2 * zoom_factor; y_half = (y_max_comp - y_min_comp)/2 * zoom_factor;

xlim_new = [x_cen - x_half, x_cen + x_half];
ylim_new = [y_cen - y_half, y_cen + y_half];

grid_skip = 1;
for i = 1:grid_skip:length(theta_x_faces)
    if theta_x_faces(i) >= xlim_new(1) && theta_x_faces(i) <= xlim_new(2)
        line([theta_x_faces(i) theta_x_faces(i)], ylim_new, 'Color', [0.88 0.88 0.92], 'LineWidth', 0.25);
    end
end
for j = 1:grid_skip:length(theta_y_faces)
    if theta_y_faces(j) >= ylim_new(1) && theta_y_faces(j) <= ylim_new(2)
        line(xlim_new, [theta_y_faces(j) theta_y_faces(j)], 'Color', [0.88 0.88 0.92], 'LineWidth', 0.25);
    end
end

xlim(xlim_new); ylim(ylim_new);

theta_circle = linspace(0, 2*pi, 300);
for j = 1:length(comp)
    if isempty(comp(j).name), continue; end
    cg = comp(j);
    
    if contains(cg.name, 'Gal') && ~contains(cg.name, 'Gas')
        plot(cg.x_c, cg.y_c, '.', 'Color', [0.4 0.4 0.8], 'MarkerSize', 6);
        continue;
    end
    
    x_circ = cg.x_c + cg.r_s * cos(theta_circle);
    y_circ = cg.y_c + cg.r_s * sin(theta_circle);
    
    if cg.is_gas
        n_rings = 5;
        for r = linspace(cg.r_s, 0, n_rings)
            x_r = cg.x_c + r * cos(theta_circle);
            y_r = cg.y_c + r * sin(theta_circle);
            alpha_r = 0.05 + 0.20*(1 - r/cg.r_s);
            if strcmp(cg.name, 'Main Gas')
                fill(x_r, y_r, c_main_gas, 'FaceAlpha', alpha_r, 'EdgeColor', 'none');
            else
                fill(x_r, y_r, c_sub_gas, 'FaceAlpha', alpha_r, 'EdgeColor', 'none');
            end
        end
        if strcmp(cg.name, 'Main Gas')
            plot(x_circ, y_circ, '-', 'Color', c_main_gas, 'LineWidth', LW.med);
        else
            plot(x_circ, y_circ, '-', 'Color', c_sub_gas, 'LineWidth', LW.med);
        end
    else
        if strcmp(cg.name, 'BCG 1')
            fill(x_circ, y_circ, c_main_gal, 'FaceAlpha', 0.35, 'EdgeColor', c_main_gal, 'EdgeAlpha', 0.9, 'LineWidth', LW.thick);
        elseif strcmp(cg.name, 'BCG 3')
            fill(x_circ, y_circ, c_sub_gal, 'FaceAlpha', 0.35, 'EdgeColor', c_sub_gal, 'EdgeAlpha', 0.9, 'LineWidth', LW.thick);
        else
            fill(x_circ, y_circ, [0.5 0.5 0.5], 'FaceAlpha', 0.35, 'EdgeColor', [0.5 0.5 0.5], 'EdgeAlpha', 0.9, 'LineWidth', LW.thick);
        end
        x_core = cg.x_c + 0.3*cg.r_s * cos(theta_circle);
        y_core = cg.y_c + 0.3*cg.r_s * sin(theta_circle);
        fill(x_core, y_core, [1 1 1], 'FaceAlpha', 0.4, 'EdgeColor', 'none');
    end
end

faint_alpha = 0.25; 
scatter(main_gal_xy(1), main_gal_xy(2), MS.l^2, c_main_gal, 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', LW.med, 'MarkerFaceAlpha', faint_alpha, 'MarkerEdgeAlpha', faint_alpha);
scatter(sub_gal_xy(1), sub_gal_xy(2), MS.m^2, c_sub_gal, 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', LW.med, 'MarkerFaceAlpha', faint_alpha, 'MarkerEdgeAlpha', faint_alpha);
scatter(main_gas_xy(1), main_gas_xy(2), MS.l^2, c_main_gas, 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', LW.med, 'MarkerFaceAlpha', faint_alpha, 'MarkerEdgeAlpha', faint_alpha);
scatter(sub_gas1_xy(1), sub_gas1_xy(2), MS.m^2, c_sub_gas, 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', LW.med, 'MarkerFaceAlpha', faint_alpha, 'MarkerEdgeAlpha', faint_alpha);

text(main_gas_xy(1), main_gas_xy(2)+comp(3).r_s+14, 'Main Gas', 'HorizontalAlignment','center','FontSize',10,'Color',c_main_gas,'FontWeight','bold');
text(sub_gas1_xy(1), sub_gas1_xy(2)+comp(4).r_s+12, 'Sub Gas 1', 'HorizontalAlignment','center','FontSize',10,'Color',c_sub_gas,'FontWeight','bold');
if ~isnan(idx_sub_gas2)
    text(sub_gas2_xy(1), sub_gas2_xy(2)+components(idx_sub_gas2).r_s/D_L*(180/pi)*3600+15, 'Sub Gas 2', 'HorizontalAlignment','center','FontSize',10,'Color',c_sub_gas);
end
if ~isnan(idx_sub_gas3)
    text(sub_gas3_xy(1), sub_gas3_xy(2)+components(idx_sub_gas3).r_s/D_L*(180/pi)*3600+8, 'Sub Gas 3', 'HorizontalAlignment','center','FontSize',10,'Color',c_sub_gas);
end

xlim([-150, 300]); ylim([-150, 200]);
xlabel('$\theta_x$ [arcsec]','Interpreter','latex','FontSize',13);
ylabel('$\theta_y$ [arcsec]','Interpreter','latex','FontSize',13);
title('Baryonic Topology --- Plummer Components \& Lensing Peaks', 'Interpreter','latex','FontSize',14,'FontWeight','bold');

h_leg_gal  = plot(NaN, NaN, 'o', 'MarkerSize', MS.m, 'MarkerFaceColor', c_main_gal, 'MarkerEdgeColor', 'k', 'LineWidth', 1);
h_leg_gas  = plot(NaN, NaN, 's', 'MarkerSize', MS.m, 'MarkerFaceColor', c_main_gas, 'MarkerEdgeColor', 'k', 'LineWidth', 1);

info_text = {'Mass Budget', sprintf('Mstars = %.2e Msun', M_obs_stars), sprintf('Mgas   = %.2e Msun', M_obs_gas), sprintf('Mtrue  = %.2e Msun', M_true_total), sprintf('eta    = %.3f', mass_ratio_eta)};
annotation('textbox', [0.72 0.3 0.18 0.28], 'String', info_text, 'FitBoxToText', 'on', 'BackgroundColor', [1 1 0.97], 'EdgeColor', [0.5 0.5 0.5], 'FontSize', 9, 'FontName', 'Helvetica', 'VerticalAlignment', 'top', 'Margin', 6, 'Interpreter', 'tex');

bar_len = 30;  bar_kpc = bar_len * kpc_per_arcsec;
bar_x = xlim_arcsec(1) + 0.06*diff(xlim_arcsec); bar_y = ylim_arcsec(1) + 0.05*diff(ylim_arcsec);
line([bar_x bar_x+bar_len], [bar_y bar_y], 'Color','k','LineWidth',2.5);
line([bar_x bar_x], [bar_y-2 bar_y+2], 'Color','k','LineWidth',2);
line([bar_x+bar_len bar_x+bar_len], [bar_y-2 bar_y+2], 'Color','k','LineWidth',2);
text(bar_x+bar_len/2, bar_y+6, sprintf('%.0f kpc', bar_kpc), 'HorizontalAlignment','center','FontSize',9,'FontWeight','bold');

filename_base = 'Fig3_BaryonicTopologyInGrid';
savefig(fig3, [filename_base, '.fig']);
print(fig3, filename_base, '-djpeg', '-r300');
fprintf('[FIGURE 3] Saved successfully.\n');

%% =====================================================================
%  FIGURE 4: MASS REVELATION CONVERGENCE κ
%% =====================================================================

fig4 = figure('Position',[100 100 900 780],'Color','w','Name','Fig4_MassRevelationKappa','PaperPositionMode','auto');

X_kpc_plot = theta_x * arcsec_to_kpc; Y_kpc_plot = theta_y * arcsec_to_kpc;

kc = min(max(kappa_mass_reveal, 1e-4), 10);
log_kc = log10(kc);

log_lo = -0.3; log_hi = 0.5;
n = 256; t = linspace(0, 1, n)';
key_colors = [0.02 0.05 0.20; 0.10 0.25 0.55; 0.25 0.60 0.80; 0.60 0.90 0.70; 1.00 1.00 1.00];
cmap_kappa = interp1(linspace(0,1,size(key_colors,1)), key_colors, t);
cmap_kappa = flipud(cmap_kappa); 

ax = axes('Position',[0.10 0.10 0.78 0.86]);
hold on; set(gca,'Color',[1 1 1]);

[THX_kpc, THY_kpc] = ndgrid(X_kpc_plot, Y_kpc_plot);
surf(THX_kpc, THY_kpc, zeros(size(THX_kpc)), log_kc, 'EdgeColor', 'none');
view(2); shading flat; axis equal tight; set(gca,'YDir','normal');

caxis([log_lo, log_hi]); colormap(ax, cmap_kappa);

bary_levels = [0.05, 0.10, 0.20];
if max(kappa_bary(:)) >= min(bary_levels)
    [Cb, hb] = contour(X_kpc_plot, Y_kpc_plot, kappa_bary', bary_levels, '--', 'Color', [0.70 0.70 0.70], 'LineWidth', 0.8, 'HandleVisibility', 'off');
    clabel(Cb, hb, 'Color', [0.55 0.55 0.55], 'FontSize', 8, 'FontWeight', 'normal', 'LabelSpacing', 500);
end

kappa_levels = [0.2, 0.5, 1.0];
if ~isempty(kappa_levels)
    [Ck, hk] = contour(X_kpc_plot, Y_kpc_plot, kappa_mass_reveal', kappa_levels, 'Color', [0.10 0.10 0.10], 'LineWidth', 1.2);
    clabel(Ck, hk, 'Color', [0.05 0.05 0.05], 'FontSize', 10, 'FontWeight', 'bold', 'LabelSpacing', 350);
end

xlim_f16_kpc = [-150, 300] * arcsec_to_kpc;
ylim_f16_kpc = [-150, 200] * arcsec_to_kpc;
xlim(xlim_f16_kpc); ylim(ylim_f16_kpc);

xlabel('x [kpc]','Interpreter','latex','FontSize',13);
ylabel('y [kpc]','Interpreter','latex','FontSize',13);
title('Mass Revelation Convergence $\kappa_{\rm Mezzi}$', 'Interpreter','latex','FontSize',14,'FontWeight','bold');

cb = colorbar('Location','eastoutside');
cb.Ticks = -0.5:0.1:0.5;
tick_labels = cell(size(cb.Ticks));
for i = 1:length(cb.Ticks)
    if cb.Ticks(i) == 0, tick_labels{i} = '$0.0$'; else, tick_labels{i} = sprintf('$%.1f$', cb.Ticks(i)); end
end
cb.TickLabels = tick_labels; cb.TickLabelInterpreter = 'latex';
cb.Label.String = '$\log_{10} \kappa$'; cb.Label.Interpreter = 'latex'; cb.Label.FontSize = 12; cb.FontSize = 10;

bar_len_kpc = 30 * arcsec_to_kpc;
bar_x = xlim_f16_kpc(1) + 0.06*diff(xlim_f16_kpc); bar_y = ylim_f16_kpc(1) + 0.05*diff(ylim_f16_kpc);
line([bar_x bar_x+bar_len_kpc], [bar_y bar_y], 'Color','k','LineWidth',2.5,'HandleVisibility','off');
line([bar_x bar_x], [bar_y-2*arcsec_to_kpc bar_y+2*arcsec_to_kpc], 'Color','k','LineWidth',2,'HandleVisibility','off');
line([bar_x+bar_len_kpc bar_x+bar_len_kpc], [bar_y-2*arcsec_to_kpc bar_y+2*arcsec_to_kpc], 'Color','k','LineWidth',2,'HandleVisibility','off');
text(bar_x+bar_len_kpc/2, bar_y+6*arcsec_to_kpc, sprintf('%.0f kpc', bar_len_kpc), 'HorizontalAlignment','center','FontSize',9,'FontWeight','bold','HandleVisibility','off');

leg_handles = []; leg_labels = {};
h_leg_true_k = plot(NaN, NaN, '-', 'Color', [0.10 0.10 0.10], 'LineWidth', 1.2);
leg_handles(end+1) = h_leg_true_k; leg_labels{end+1} = 'True frame $\kappa$ ';
h_leg_bary = plot(NaN, NaN, '--', 'Color', [0.70 0.70 0.70], 'LineWidth', 1.0);
leg_handles(end+1) = h_leg_bary; leg_labels{end+1} = 'Baryonic baseline $\kappa$';
legend(leg_handles, leg_labels, 'Location', 'north', 'FontSize', 9, 'Box', 'on', 'EdgeColor', [0.6 0.6 0.6], 'Interpreter', 'latex');
set(gca, 'FontSize', 11, 'Box', 'on', 'LineWidth', 0.8, 'Layer', 'top');

% INSET: BCG3 Zoom
bcg1_kpc = main_gal_xy * arcsec_to_kpc;
bcg2_kpc = comp_xy_arcsec(idx_bcg2, :) * arcsec_to_kpc;
bcg3_kpc = sub_gal_xy * arcsec_to_kpc;

zoom_half = 30; 
zoom_xlim = [bcg3_kpc(1) - zoom_half, bcg3_kpc(1) + zoom_half];
zoom_ylim = [bcg3_kpc(2) - zoom_half, bcg3_kpc(2) + zoom_half];

axes(ax); 
rectangle('Position', [zoom_xlim(1), zoom_ylim(1), diff(zoom_xlim), diff(zoom_ylim)], 'EdgeColor', [0.85 0.20 0.20], 'LineWidth', 1.5, 'LineStyle', '-');

ax_inset = axes('Position', [0.55 0.19 0.24 0.24], 'Color', [1 1 1], 'Box', 'on', 'LineWidth', 1.0);
hold(ax_inset, 'on');

surf(ax_inset, THX_kpc, THY_kpc, zeros(size(THX_kpc)), log_kc, 'EdgeColor', 'none');
view(ax_inset, 2); shading(ax_inset, 'flat'); set(ax_inset, 'YDir', 'normal');
caxis(ax_inset, [log_lo, log_hi]); colormap(ax_inset, cmap_kappa);
xlim(ax_inset, zoom_xlim); ylim(ax_inset, zoom_ylim);

if max(kappa_bary(:)) >= min(bary_levels)
    contour(ax_inset, X_kpc_plot, Y_kpc_plot, kappa_bary', bary_levels, '--', 'Color', [0.70 0.70 0.70], 'LineWidth', 0.7);
end
if ~isempty(kappa_levels)
    [Ck_in, hk_in] = contour(ax_inset, X_kpc_plot, Y_kpc_plot, kappa_mass_reveal', kappa_levels, 'Color', [0.10 0.10 0.10], 'LineWidth', 1.0);
    clabel(Ck_in, hk_in, 'Color', [0.05 0.05 0.05], 'FontSize', 8, 'FontWeight', 'bold', 'LabelSpacing', 250);
end

plot(ax_inset, bcg3_kpc(1), bcg3_kpc(2), 'r+', 'MarkerSize', 10, 'LineWidth', 1.5, 'HandleVisibility', 'off');
title(ax_inset, 'BCG3 Zoom (60 kpc)', 'FontSize', 9, 'FontWeight', 'bold');
xlabel(ax_inset, 'x [kpc]', 'FontSize', 8); ylabel(ax_inset, 'y [kpc]', 'FontSize', 8);
set(ax_inset, 'FontSize', 8, 'Layer', 'top', 'XColor', [0.3 0.3 0.3], 'YColor', [0.3 0.3 0.3]);

filename_base = 'Fig4_MassRevelationKappa';
print(fig4, filename_base, '-djpeg', '-r300');
fprintf('[FIGURE 4] Saved successfully.\n');


%% =====================================================================
%  FIGURE 5: COMPONENT-LEVEL MASS COMPARISON
%% =====================================================================

fig5 = figure('Color', 'white', 'Units', 'inches', 'Position', [1, 1, 9.5, 6.5], 'PaperUnits', 'inches', 'PaperPosition', [0 0 9.5 6.5], 'Renderer', 'painters');
ax_main = axes('Position', [0.10, 0.12, 0.75, 0.82]);
hold on;

numComp = length(components);
M_obs_all = zeros(numComp + 1, 1); M_true_all = zeros(numComp + 1, 1); X_pos_all = zeros(numComp + 1, 1); comp_names_all = cell(numComp + 1, 1);

[DX, DY] = ndgrid(dx_fine, dy_fine);
dA_obs_pc2 = (DX/pc) .* (DY/pc);

x_kpc_temp = x_fine / (pc * 1e3); y_kpc_temp = y_fine / (pc * 1e3);
F_Jmap = griddedInterpolant({x_kpc_temp, y_kpc_temp}, J, 'linear', 'none');

fprintf('\n[FIGURE 5] Calculating component masses...\n');

for j = 1:numComp
    comp = components(j);
    M_obs = abs(comp.M_total) / Msun; 
    if M_obs == 0, continue; end
    
    M_true_val = 0;
    is_compact_galaxy = contains(comp.name, 'Gal') && ~contains(comp.name, 'Gas');
    
    if is_compact_galaxy
        xc_kpc = comp.x_c / (pc * 1e3); yc_kpc = comp.y_c / (pc * 1e3);
        J_at_comp = F_Jmap(xc_kpc, yc_kpc);
        M_true_val = M_obs * J_at_comp;
    else
        Xs_f = X_FINE - comp.x_c; Ys_f = Y_FINE - comp.y_c;
        R2_f = Xs_f.^2 + Ys_f.^2;
        Sigma_obs_j_SI = (comp.M_total .* comp.r_s^2) ./ (pi .* (R2_f + comp.r_s^2).^2);
        Sigma_obs_j_Msun_pc2 = Sigma_obs_j_SI .* (pc^2 / Msun);
        M_true_val = sum(Sigma_obs_j_Msun_pc2(:) .* J(:) .* dA_obs_pc2(:));
    end
    M_true_val = max(M_true_val, M_obs);
    M_obs_all(j) = M_obs; M_true_all(j) = M_true_val; X_pos_all(j) = comp.x_c / (pc * 1e3); comp_names_all{j} = comp.name;
end

if exist('M_obs_total', 'var') && exist('M_true_total', 'var')
    M_obs_all(end) = M_obs_total; M_true_all(end) = M_true_total;
    valid_comp_idx = M_obs_all(1:end-1) > 0;
    X_pos_all(end) = mean(X_pos_all(valid_comp_idx)); 
    comp_names_all{end} = 'Overall Cluster';
end

valid_idx = M_obs_all > 0 & M_true_all > 0;
M_obs_valid = M_obs_all(valid_idx); M_true_valid = M_true_all(valid_idx); X_pos_valid = X_pos_all(valid_idx); comp_names_valid = comp_names_all(valid_idx);

[X_pos_sorted, sort_idx] = sort(X_pos_valid, 'ascend');
log_M_obs_sorted = log10(M_obs_valid(sort_idx)); log_M_true_sorted = log10(M_true_valid(sort_idx));
comp_names_sorted = comp_names_valid(sort_idx); correction_sorted = M_true_valid(sort_idx) ./ M_obs_valid(sort_idx);
numPts = length(log_M_obs_sorted);

log_M_obs_sorted = log_M_obs_sorted(:); log_M_true_sorted = log_M_true_sorted(:); correction_sorted = correction_sorted(:);

cmap = turbo(256);
corr_min_log = 0; corr_max_log = 1.7;
log_correction = log10(correction_sorted);

for g = 1:numPts
    norm_val = (log_correction(g) - corr_min_log) / (corr_max_log - corr_min_log);
    idx = max(1, min(256, round(norm_val * 255) + 1));
    pt_color = cmap(idx, :);
    plot([X_pos_sorted(g), X_pos_sorted(g)], [log_M_obs_sorted(g), log_M_true_sorted(g)], '-', 'Color', [pt_color, 0.15], 'LineWidth', 6, 'HandleVisibility', 'off');
    plot([X_pos_sorted(g), X_pos_sorted(g)], [log_M_obs_sorted(g), log_M_true_sorted(g)], '-', 'Color', pt_color, 'LineWidth', 2.5, 'HandleVisibility', 'off');
end

edge_color = [0.15 0.15 0.15]; edge_width = 1.2; marker_size = 80;
h_obs = scatter(X_pos_sorted, log_M_obs_sorted, marker_size, [0.85 0.85 0.85], 'o', 'filled', 'MarkerEdgeColor', edge_color, 'LineWidth', edge_width, 'DisplayName', 'Observed Mass');
h_true = scatter(X_pos_sorted, log_M_true_sorted, marker_size, log_correction, '^', 'filled', 'MarkerEdgeColor', edge_color, 'LineWidth', edge_width, 'DisplayName', 'True Mass');

colormap(ax_main, cmap); caxis([corr_min_log, corr_max_log]); 
cbar = colorbar('Location', 'eastoutside', 'Position', [0.88, 0.12, 0.03, 0.82]);
cbar.Label.String = 'M_{true}/M_{obs}'; cbar.Label.FontSize = 18; cbar.Label.FontWeight = 'bold'; cbar.Label.Interpreter = 'tex';
cbar.Ticks = [log10(1), log10(5), log10(10), log10(20), log10(50)]; cbar.TickLabels = {'1x', '5x', '10x', '20x', '50x'}; cbar.FontSize = 10;

for g = 1:numPts
    name = comp_names_sorted{g};
    if contains(name, 'Gas') || contains(name, 'BCG') || strcmp(name, 'Overall Cluster')
        y_pos = log_M_true_sorted(g); x_pos = X_pos_sorted(g);
        if strcmp(name, 'Subcluster Gas 2'), y_pos = y_pos + 0.15; end
        if strcmp(name, 'Overall Cluster'), y_pos = y_pos - 0.3; end
        text(x_pos + 25, y_pos, name, 'FontSize', 9, 'Color', 'k', 'FontWeight', 'bold', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'BackgroundColor', [1 1 1 0.0], 'EdgeColor', 'none', 'Margin', 1);
    end
end

xlabel('Component X position [kpc]', 'FontSize', 18, 'FontWeight', 'bold', 'Interpreter', 'tex');
ylabel('log_{10}(M/M_{\odot})', 'FontSize', 18, 'FontWeight', 'bold', 'Interpreter', 'tex');
legend([h_obs, h_true], {'Observed Mass', 'True Mass'}, 'Location', 'northwest', 'FontSize', 11, 'Box', 'on', 'EdgeColor', [0.3 0.3 0.3], 'LineWidth', 1);

all_y_values = [log_M_obs_sorted(:); log_M_true_sorted(:)];
y_min = min(all_y_values); y_max = max(all_y_values); y_pad = (y_max - y_min) * 0.08;
set(ax_main, 'FontSize', 10, 'LineWidth', 1.2, 'Box', 'on', 'XLim', [-670, 1250], 'YLim', [y_min - y_pad, y_max + y_pad], 'XTick', -600:200:1200, 'YGrid', 'on', 'XGrid', 'off', 'GridAlpha', 0.15, 'Layer', 'top', 'Color', 'white');

filename_base = 'Fig5_MassComparison_Spatial';
savefig(fig5, [filename_base, '.fig']);
print(fig5, filename_base, '-djpeg', '-r300');
fprintf('[FIGURE 5] Saved successfully.\n');

%% ------------------------------------------------------------------------
%  SECTION 9: SAVE RESULTS
% ------------------------------------------------------------------------
fprintf('\n[SAVE] Compiling and saving results to .mat file...\n');

% Initialize a clean, nested struct for organized storage
Results = struct();

% 1. Configuration & Cosmology
Results.Config = struct( ...
    'Compliance', Compliance, ...
    'Sigma_smoothing', Sigma_smoothing, ...
    'smoothing_fwhm_arcsec', smoothing_fwhm_arcsec, ...
    'Box_3d', Box_3d, ...
    'N_2d', N_2d, ...
    'N_z', N_z, ...
    'stretch_factor', stretch_factor, ...
    'N_fft', N_fft, ...
    'fft_box_half_size_kpc', fft_box_half_size_kpc, ...
    'N_theta', N_theta, ...
    'N_phi', N_phi, ...
    'N_r_ray', N_r_ray ...
);

Results.Cosmology = struct( ...
    'z_lens', z_lens, ...
    'D_L', D_L, ...
    'Sigma_crit', Sigma_crit, ...
    'arcsec_to_kpc', arcsec_to_kpc ...
);

% 2. Mass Model Data
Results.MassModel = struct( ...
    'components', components, ...
    'star_idx', star_idx, ...
    'gas_idx', gas_idx, ...
    'comp_xy_kpc', comp_xy_kpc, ...
    'comp_xy_arcsec', comp_xy_arcsec, ...
    'M_main_smooth', M_main_smooth, ...
    'M_sub_smooth', M_sub_smooth ...
);

% 3. Grid Data
Results.Grid = struct( ...
    'x_fine', x_fine, ...
    'y_fine', y_fine, ...
    'z_axis', z_axis, ...
    'theta_x', theta_x, ...
    'theta_y', theta_y, ...
    'dx_fine', dx_fine, ...
    'dy_fine', dy_fine, ...
    'dz_3d', dz_3d ...
);

% 4. Mezzi Ray Tracing Results
Results.Mezzi = struct( ...
    'zeta_2D_fine', zeta_2D_fine, ...
    'J', J, ...
    'lambda_2D', lambda_2D ...
);

% 5. Lensing & Convergence Maps
Results.Lensing = struct( ...
    'Sigma_stars_fine', Sigma_stars_fine, ...
    'Sigma_gas_fine', Sigma_gas_fine, ...
    'Sigma_true_stars', Sigma_true_stars, ...
    'Sigma_true_gas', Sigma_true_gas, ...
    'M_obs_total', M_obs_total, ...
    'M_true_total', M_true_total, ...
    'mass_ratio_eta', mass_ratio_eta, ...
    'kappa_bary', kappa_bary, ...
    'kappa_mass_reveal', kappa_mass_reveal, ...
    'kappa_curv_amp', kappa_curv_amp, ...
    'models', models ...
);

% Save to file
save_filename = 'Mezzi_BulletCluster_Results.mat';
save(save_filename, 'Results');

fprintf('[SAVE] Results successfully saved to %s\n\n', save_filename);


%% ------------------------------------------------------------------------
% SECTION 10: LOCAL FUNCTIONS
% ------------------------------------------------------------------------
function [K_3D, Zeta_3D] = trace_component_K(comp, Phi, M_input, X_3D, Y_3D, Z_3D, r_ray, R_box, c_speed, Compliance, N_theta, N_phi)
    G = 6.67430e-11;
    xc = comp.x_c; yc = comp.y_c; zc = comp.z_c;
    Xs = X_3D - xc; Ys = Y_3D - yc; Zs = Z_3D - zc;
    R_sph = sqrt(Xs.^2 + Ys.^2 + Zs.^2);
    
    R_flat = R_sph(:); X_flat = Xs(:); Y_flat = Ys(:); Z_flat = Zs(:);
    
    theta_bin = linspace(0, pi, N_theta+1); phi_bin = linspace(0, 2*pi, N_phi+1);
    theta_pts = acos(max(min(Z_flat ./ max(R_flat,1e-10), 1), -1));
    phi_pts   = mod(atan2(Y_flat, X_flat), 2*pi);
    t_idx = discretize(theta_pts, theta_bin); p_idx = discretize(phi_pts, phi_bin);
    dir_idx = sub2ind([N_theta, N_phi], t_idx, p_idx);
    
    invalid_mask = isnan(dir_idx) | dir_idx == 0; dir_idx(invalid_mask) = 1;
    N_dirs = N_theta * N_phi;
    
    n_x = accumarray(dir_idx, X_flat, [N_dirs, 1], @mean); n_y = accumarray(dir_idx, Y_flat, [N_dirs, 1], @mean); n_z = accumarray(dir_idx, Z_flat, [N_dirs, 1], @mean);
    norm_n = sqrt(n_x.^2 + n_y.^2 + n_z.^2);
    empty_bins = norm_n == 0; n_x(empty_bins) = 1; n_y(empty_bins) = 0; n_z(empty_bins) = 0; norm_n(empty_bins) = 1;
    n_x = n_x ./ norm_n; n_y = n_y ./ norm_n; n_z = n_z ./ norm_n;
    
    x_ray = r_ray .* n_x'; y_ray = r_ray .* n_y'; z_ray = r_ray .* n_z';
    Phi_ray = Phi(x_ray + xc, y_ray + yc, z_ray + zc); 
    
    outside = r_ray > R_box; tail_vals = -G * M_input ./ r_ray(outside); 
    Phi_ray(outside, :) = repmat(tail_vals, 1, size(Phi_ray, 2));
    
    safe_Phi = max(abs(Phi_ray), 1e-10);
    integrand = (Phi_ray ./ safe_Phi) .* sqrt(2 .* abs(Phi_ray)) ./ (c_speed .* r_ray);
    cum_int = cumtrapz(r_ray, integrand, 1); total_int = cum_int(end, :); 
    
    [~, idx_lo] = histc(R_flat, r_ray); idx_lo(idx_lo < 1) = 1; idx_lo(idx_lo >= length(r_ray)) = length(r_ray)-1; idx_hi = idx_lo + 1;
    r_lo = r_ray(idx_lo); r_hi = r_ray(idx_hi);
    denom = (r_hi - r_lo); denom(denom == 0) = 1; w = (R_flat - r_lo) ./ denom;
    
    lin_idx_lo = sub2ind(size(cum_int), idx_lo, dir_idx); lin_idx_hi = sub2ind(size(cum_int), idx_hi, dir_idx);
    val_lo = cum_int(lin_idx_lo); val_hi = cum_int(lin_idx_hi);
    cum_at_R = val_lo + w .* (val_hi - val_lo);
    
    overshoot = R_flat >= r_ray(end);
    if any(overshoot), cum_at_R(overshoot) = cum_int(end, dir_idx(overshoot))'; end
    
    CumP = total_int(dir_idx)' - cum_at_R; 
    CumP(R_flat > r_ray(end)) = 0;
    K_2D = exp(CumP);
    
    K_3D = reshape(K_2D, size(X_3D)); 
    Zeta_3D = exp(Compliance * (K_3D - 1));
end

function x_faces = generate_dual_focus_faces(u_faces, Box_half, x1, x2, sigma)
    L = Box_half;
    baseline = 0.05; 
    u_norm = (u_faces + 1) / 2;
    x_dense = linspace(-L, L, 500000);
    
    term1 = (sigma * sqrt(pi)/2) * (erf((x_dense - x1)/sigma) - erf((-L - x1)/sigma));
    term2 = (sigma * sqrt(pi)/2) * (erf((x_dense - x2)/sigma) - erf((-L - x2)/sigma));
    term3 = baseline * (x_dense - (-L));
    
    cdf = term1 + term2 + term3;
    cdf = cdf / cdf(end); 
    x_faces = interp1(cdf, x_dense, u_norm, 'linear', 'extrap');
end

function [x, y, z, r] = sampleflattened_famaey(N, a, x0, y0, seed, qxyz, PAdeg)
    rng(seed);
    u = rand(1, N);
    r = a ./ sqrt(u.^(-2.0/3.0) - 1.0); 
    costh = rand(1, N) * 2.0 - 1.0;
    sinth = sqrt(1.0 - costh.^2);
    phi = rand(1, N) * 2.0 * pi;
    nx = sinth .* cos(phi); ny = sinth .* sin(phi); nz = costh;

    qx = qxyz(1); qy = qxyz(2); qz = qxyz(3);
    sx = qx * nx; sy = qy * ny; sz = qz * nz;
    norm = sqrt(sx.^2 + sy.^2 + sz.^2);
    nx2 = sx ./ norm; ny2 = sy ./ norm; nz2 = sz ./ norm;
    
    pa = deg2rad(PAdeg);
    cp = cos(pa); sp = sin(pa);
    rx = cp * nx2 - sp * ny2;
    ry = sp * nx2 + cp * ny2;
    rz = nz2;
    
    x = x0 + r .* rx; y = y0 + r .* ry; z = r .* rz;
end

function [x, y, z] = snaptotargets_famaey(x, y, z, rorig, targets, cx, cy)
    N = length(x);
    available = true(1, N);
    for t = 1:size(targets, 1)
        xt = targets(t, 1); yt = targets(t, 2);
        if ~any(available), break; end
        idxs = find(available);
        d2 = (x(idxs) - xt).^2 + (y(idxs) - yt).^2;
        [~, min_idx] = min(d2);
        j = idxs(min_idx);
        available(j) = false;
        x(j) = xt; y(j) = yt;
        rho2 = (xt - cx)^2 + (yt - cy)^2;
        z2 = rorig(j)^2 - rho2;
        if z2 >= 0.0
            if z(j) >= 0.0, z(j) = sqrt(z2); else, z(j) = -sqrt(z2); end
        else
            z(j) = 0.0;
        end
    end
end