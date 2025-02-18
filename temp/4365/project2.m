%% Exercise 6

disp(23+10^37+1000-10^37)
%% Exercise 7

A = magic(5);
A(:,4)
A(2,:)
A(1:2:5,:)
A(2:4,2:4)
A(3,5)
A([1,5],[1,5])

%% Exercise 8

a = 5;
b = 2;
c = -1;

disp(a < 10 & a > 0)
disp(a < 4 | ( c <= 2 & b>2 ))
disp(c ~= -1)
disp(b == a + 3*c)
disp(abs(c) - b/2 > 0)

%% Exercise 10

% part a
t = linspace(0, 2*pi, 10000);
x = cos(t);
y = sin(t);
z = cos(2*t);

figure10a = plot3(x,y,z, linewidth=2);
exportgraphics(gcf,'figure10a.pdf','ContentType','vector')
clf('reset')

% part b
[X,Y,Z] = peaks(30);
figure10b1 = mesh(X,Y,Z);
exportgraphics(gcf,'figure10b1.pdf','ContentType','vector')
clf('reset')

figure10b2 = plot3(X,Y,Z);
exportgraphics(gcf,'figure10b2.pdf','ContentType','vector')
clf('reset')

figure10b3 = surf(X,Y,Z);
exportgraphics(gcf,'figure10b3.pdf','ContentType','vector')
clf('reset')

% part c
[X,Y] = meshgrid(linspace(-0.5,0.5,250));
Z = log(1./(sqrt(X.^2 + Y.^2)));

figure10c = mesh(X,Y,Z);
exportgraphics(gcf,'figure10c.pdf','ContentType','vector')
clf('reset')

%% Exercise 11

m = 10;
n = randi([3,30],1,m);
midpoints = 10*rand(2,m);
colors = randi([0,255],3,m)/255;

principal_angles = 2*pi./n;

t = linspace(0,2*pi,1000);

for i = 1:m
    angles = (0:(n(i)-1))*principal_angles(i);
    points = zeros(2,n(i));
    for j = 1:n(i)
        points(:,j) = [cos(angles(j)) + midpoints(1,i);sin(angles(j)) + ...
            midpoints(2,i)];
    end
    patch(points(1,:),points(2,:),colors(:,i)')
    hold on
end

axis equal
exportgraphics(gcf,'figure11.pdf','ContentType','vector')
clf('reset')