fac= 100;
lc1 = 0.0005;
xmax = 5/fac;
ymax = 2/fac;

Point(1) = {0,0,0,lc1};

Point(2) = {xmax*1.75/3, 0, 0, lc1};
Point(3) = {xmax*1.75/3, ymax/6, 0, lc1}; 

Point(4) = {xmax*1.85/3, ymax/6, 0,lc1}; 
Point(5) = {xmax*1.85/3,0,0,lc1};

Point(6) = {xmax,0,0,lc1};
Point(7) = {xmax,ymax,0,lc1};
Point(8) = {0,ymax,0,lc1};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,5};
Line(5) = {5,6};
Line(6) = {6,7};
Line(7) = {7,8};
Line(8) = {8,1};

Line Loop(1) = {1:8};
Plane Surface(1) = {1};
 
// Transfinite Surface {1};
// Recombine Surface {1};
 
Physical Line(101) = {1,2,3,4,5};
// Physical Line(102) = {2};
// Physical Line(103) = {3};
// Physical Line(104) = {4};
// Physical Line(105) = {5};
Physical Line(102) = {6};
Physical Line(103) = {7};
Physical Line(104) = {8};
Physical Surface(150)={1};
