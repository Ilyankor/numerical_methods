clear all; close all;

% solution plot
epsilon = 0.01;
betavec = [0.2 2 20 200];
n = 100;

figure(1)
for ii=1:length(betavec)
    subplot(2,2,ii);
    beta = betavec(ii);

    % central differences
    [x,ucd] = ADVECTION_DIFFUSION(epsilon,beta,n,1);
    % upwind
    [x,uup] = ADVECTION_DIFFUSION(epsilon,beta,n,2);
    % Scharfetter-Gummel
    [x,usg] = ADVECTION_DIFFUSION(epsilon,beta,n,3);
    % exact solution
    uex = (exp(beta/epsilon*(x-1))-exp(-beta/epsilon))./(1-exp(-beta/epsilon));

    plot(x,uex,'r-','LineWidth',3);
    hold on; box on;
    plot(x,ucd,'b:','LineWidth',3);
    plot(x,uup,'k-.','LineWidth',3);
    plot(x,usg,'m--','LineWidth',3);
    xlabel('x','FontSize',20,'Color','k');
    ylabel('u(x)','FontSize',20,'Color','k');
    title(strcat('\beta=',num2str(beta)),'FontSize',20);
    set(gca,'FontSize',20);
end
subplot(2,2,1);
legend('exact','central','upwind','SG','FontSize',12);

% error plot
epsilon = 0.01;
beta = 0.2;
maxstep = 13;

errcd = zeros(1,maxstep);
errup = zeros(1,maxstep);
errsg = zeros(1,maxstep);
nvec = 2.^(1:maxstep);

for k=1:maxstep
    n = nvec(k);
    % central differences
    [x,ucd] = ADVECTION_DIFFUSION(epsilon,beta,n,1);
    % upwind
    [x,uup] = ADVECTION_DIFFUSION(epsilon,beta,n,2);
    % Scharfetter-Gummel
    [x,usg] = ADVECTION_DIFFUSION(epsilon,beta,n,3);
    
    % evaluate the error also at intermediate steps
    neval = n*10;
    xfine = linspace(0,1,neval+1);
    % exact solution
    uexfine = (exp(beta/epsilon*xfine)-1)./(exp(beta/epsilon)-1);
    % linear interpolation of the numerical solutions
    ucdfine = interp1(x,ucd,xfine);
    uupfine = interp1(x,uup,xfine);
    usgfine = interp1(x,usg,xfine);
    
    % calculate the errors
    errcd(k) = norm(ucdfine-uexfine,inf);
    errup(k) = norm(uupfine-uexfine,inf);
    errsg(k) = norm(usgfine-uexfine,inf);
end

figure(2);
loglog(nvec,errcd,'b-','LineWidth',3);
hold on; box on; grid on;
loglog(nvec,errup,'k-.','LineWidth',3);
loglog(nvec,errsg,'m--','LineWidth',3);
xlabel('number of steps','FontSize',20,'Color','k');
ylabel('error','FontSize',20,'Color','k');
set(gca,'FontSize',20);
xlim([2 10000]);
legend('central','upwind','SG');
