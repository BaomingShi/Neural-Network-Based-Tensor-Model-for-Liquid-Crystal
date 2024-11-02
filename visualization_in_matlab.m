xlin = linspace(-1,1,100);

ylin = linspace(-1,1,100);

zlin = linspace(-1,1,100);
[X,Y,Z] = meshgrid(xlin,ylin,zlin);
E1 = diag([1,0,-1]);E2 = [0,0,0;0,1,0;0,0,-1];E3 = [0,1,0;1,0,0;0,0,0];
E4 = [0 0 1 ; 0 0 0 ;1 0 0 ];E5=[0 0 0 ; 0 0 1; 0 1 0];



load test_shell.mat

central_inner=-0.1;
central_radius=0.7;



q1=double(q1);q2=double(q2);q3=double(q3);q4=double(q4);q5=double(q5);
point_total=double(point_total);

bbeta=zeros(size(q1));Maxeig=zeros(size(q1));
u=zeros(size(q1));v=zeros(size(q1));w=zeros(size(q1));
for i=1:length(point_total)
    Q  = q1(i) * E1 + q2(i) * E2 + q3(i) * E3+q4(i)*E4+q5(i)*E5;
    [V, D] = eig(Q);

    ix = 3;
    if D(2,2) > D(3,3)
        ix = 2;
    end

    if D(1,1) > D(ix, ix)
        ix = 1;
    end
    u(i) = V(1,ix);
    v(i) = V(2,ix);
    w(i) = V(3,ix);
    trQ2 = 0;
    trQ3 = 0;
    for mm = 1:3
        trQ2 = trQ2 + D(mm,mm)^2;
        trQ3 = trQ3 + D(mm,mm)^3;
    end
    Maxeig(i) = D(ix, ix);
    if max(trQ2,abs(trQ3))<0.02
        bbeta(i) = 0;
    else
        bbeta(i) = 1-6*trQ3^2/trQ2^3;
    end
end



xx=point_total(:,1);yy=point_total(:,2);zz=point_total(:,3);
F = scatteredInterpolant(xx,yy,zz,bbeta);
bbeta = F(X,Y,Z);
bbeta(X.^2+Y.^2+Z.^2>=1)=NaN;
bbeta(X.^2+Y.^2+(Z-central_inner).^2<central_radius^2)=NaN;
pcolor3D(X,Y,Z,bbeta)
colorbar;
shading interp;

hold on




N=length(xx);
randposition = randperm(N);

NN=1000;%%Larger，more vector
randposition = randposition(1:NN);
qH = quiver3(xx(randposition),yy(randposition),zz(randposition),u(randposition).*Maxeig(randposition),v(randposition).*Maxeig(randposition),w(randposition).*Maxeig(randposition),1.2,'w','linewidth',1);
set(qH,'ShowArrowHead','off')
axis off
colormap([[linspace(0.05,0.30,20) linspace(0.31,0.70,20) linspace(0.70,0.80,60)  linspace(0.80,0.66,50)]' ...
    [linspace(0.1,0.40,20) linspace(0.41,0.70,20) linspace(0.70,0.41,60)  linspace(0.40,0.20,50)]' ...
    [linspace(0.6,0.6,20) linspace(0.70,0.70,20) linspace(0.70,0.31,60)  linspace(0.30,0.15,50)]']);

fig = gcf;fig.PaperPositionMode = 'auto';fig_pos = fig.PaperPosition;fig.PaperSize = [fig_pos(3) fig_pos(4)];
shading interp

h=colorbar;
xlabel('x','Interpreter','latex')
ylabel('y','Interpreter','latex')
zlabel('z','Interpreter','latex')

set(gca,'FontSize', 20);

set(h, 'FontSize', 30);

caxis([0,1])
axis equal
hold off
axis off

